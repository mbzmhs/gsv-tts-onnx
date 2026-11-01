/* api.c - public C API implementation (lifecycle, state, inference) */
#include "engine.h"
#include "gpt.h"
#include "sovits_dec.h"
#include "g2p_dispatch.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#define GSV_VERSION_STR "0.2.0"

static HMODULE g_self_module = NULL;

BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) g_self_module = h;
    return TRUE;
}

/* ---------------- global error (engine is single-threaded by contract) ---------------- */
static char t_global_err[512];

const char *gsv_global_last_error(void) {
    return t_global_err;
}

void gsv_set_error(gsv_engine *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(t_global_err, sizeof(t_global_err), fmt, ap);
    if (e) snprintf(e->last_error, sizeof(e->last_error), "%s", t_global_err);
    va_end(ap);
}

const char *gsv_last_error(const gsv_engine *e) {
    return e ? e->last_error : "";
}

const char *gsv_version(void) {
    return GSV_VERSION_STR;
}

/* ---------------- config parsing (minimal JSON, flat strings) ---------------- */
static void json_get_str(const char *json, const char *key, char *out, size_t n) {
    if (!json || !out) return;
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == '"') {
        p++;
        size_t i = 0;
        while (*p && *p != '"' && i + 1 < n) {
            if (*p == '\\' && p[1]) p++;
            out[i++] = *p++;
        }
        out[i] = 0;
    } else {
        snprintf(out, n, "%s", p);
    }
}

/* returns 1 if key found and parsed */
static int json_get_int(const char *json, const char *key, int *out) {
    if (!json || !out) return 0;
    char pat[128], val[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    snprintf(val, sizeof(val), "%s", p);
    *out = atoi(val);
    return 1;
}

/* returns 1 if key found and parsed */
static int json_get_float(const char *json, const char *key, float *out) {
    if (!json || !out) return 0;
    char pat[128], val[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    snprintf(val, sizeof(val), "%s", p);
    *out = (float)atof(val);
    return 1;
}

/* ---------------- ORT bootstrap ---------------- */
static void build_ort_path(gsv_engine *e, char *out, size_t n) {
    if (e->ort_lib_dir[0]) {
        snprintf(out, n, "%s\\onnxruntime.dll", e->ort_lib_dir);
        return;
    }
    char dir[512] = "";
    if (g_self_module) {
        GetModuleFileNameA(g_self_module, dir, sizeof(dir));
        char *slash = strrchr(dir, '\\');
        if (slash) *slash = 0;
    }
    if (!dir[0]) GetCurrentDirectoryA(sizeof(dir), dir);
    snprintf(out, n, "%s\\onnxruntime.dll", dir);
}

int gsv_ort_bootstrap(gsv_engine *e) {
    char path[600];
    build_ort_path(e, path, sizeof(path));
    e->ort_module = LoadLibraryA(path);
    if (!e->ort_module) {
        /* fall back to plain name (PATH / system) */
        e->ort_module = LoadLibraryA("onnxruntime.dll");
    }
    if (!e->ort_module) {
        gsv_set_error(e, "LoadLibrary(onnxruntime.dll) failed: %lu (tried %s)",
                      GetLastError(), path);
        return -1;
    }
    typedef const OrtApiBase *(ORT_API_CALL *GetApiBaseFn)(void);
    GetApiBaseFn get_base = (GetApiBaseFn)GetProcAddress(e->ort_module, "OrtGetApiBase");
    if (!get_base) {
        gsv_set_error(e, "OrtGetApiBase not found in onnxruntime.dll");
        return -1;
    }
    e->api_base = get_base();
    if (!e->api_base || !e->api_base->GetApi) {
        gsv_set_error(e, "OrtApiBase invalid");
        return -1;
    }
    e->api = e->api_base->GetApi(ORT_API_VERSION);
    if (!e->api) {
        gsv_set_error(e, "ORT API version %d unsupported by onnxruntime.dll",
                      ORT_API_VERSION);
        return -1;
    }
    return 0;
}

/* ---------------- lifecycle ---------------- */
gsv_engine *gsv_engine_create(const char *config_json) {
    gsv_engine *e = (gsv_engine *)calloc(1, sizeof(gsv_engine));
    if (!e) {
        snprintf(t_global_err, sizeof(t_global_err), "out of memory");
        return NULL;
    }
    pthread_mutex_init(&e->infer_lock, NULL);
    snprintf(e->device, sizeof(e->device), "%s", "auto");
    e->intra_op_threads = 0;
    e->use_fp16 = -1;   /* auto: cuda prefers *_fp16.onnx when present */
    gsv_sampling_defaults(&e->sampling);
    e->sentence_pause = 0.3f;   /* legacy inter-sentence silence (streaming path only) */
    e->cut_mute = 0.3f;         /* chunk-boundary silence base, seconds (÷speed, ×tail scale) */
    e->cut_minlen = 0.0f;       /* merge clauses until this many semantic chars (0 = every punctuation) */
    /* default RNG seed (deterministic across runs); override via "rng_seed" */
    e->rng_state = 0x9E3779B97F4A7C15ULL;

    char base_manifest[512] = "";
    if (config_json && config_json[0]) {
        json_get_str(config_json, "device", e->device, sizeof(e->device));
        json_get_int(config_json, "intra_op_num_threads", &e->intra_op_threads);
        int fp16 = -1;
        if (json_get_int(config_json, "use_fp16", &fp16)) e->use_fp16 = fp16;
        json_get_str(config_json, "ort_lib_dir", e->ort_lib_dir, sizeof(e->ort_lib_dir));
        json_get_str(config_json, "base_manifest", base_manifest, sizeof(base_manifest));
        int seed = 0;
        if (json_get_int(config_json, "rng_seed", &seed)) {
            e->rng_state = (uint64_t)(uint32_t)seed;
            if (e->rng_state == 0) e->rng_state = 1;   /* xorshift needs non-zero */
        }
        float sp = -1.0f;
        if (json_get_float(config_json, "sentence_pause", &sp) && sp >= 0.0f) e->sentence_pause = sp;
        float cm = -1.0f;
        if (json_get_float(config_json, "cut_mute", &cm) && cm >= 0.0f) e->cut_mute = cm;
        float cml = -1.0f;
        if (json_get_float(config_json, "cut_minlen", &cml) && cml >= 0.0f) e->cut_minlen = cml;
    }

    if (gsv_ort_bootstrap(e) != 0) {
        pthread_mutex_destroy(&e->infer_lock);
        free(e);
        return NULL;
    }

    /* optional base model package: loaded once at init from the per-version manifest.
     * If present and it fails to load (bad version / missing models), create fails. */
    if (base_manifest[0]) {
        int brc = gsv_load_base(e, base_manifest);
        if (brc != GSV_OK) {
            snprintf(t_global_err, sizeof(t_global_err), "base load failed: %s", gsv_last_error(e));
            gsv_engine_destroy(e);
            return NULL;
        }
    }
    return e;
}

void gsv_engine_destroy(gsv_engine *e) {
    if (!e) return;
    gsv_engine_release_models(e);   /* character + base sessions, VQ, G2P, BERT vocab */
    if (e->env) { e->api->ReleaseEnv(e->env); e->env = NULL; }
    if (e->ort_module) FreeLibrary(e->ort_module);
    pthread_mutex_destroy(&e->infer_lock);
    free(e);
}

/* ---------------- stubs (implemented in later milestones) ---------------- */
/* gsv_load_model / gsv_unload_model live in engine.c */

/* gsv_warmup_ref / gsv_warmup_refs / gsv_unload_ref live in refs.c */

int gsv_get_state(const gsv_engine *e, char *buf, int buf_size, int *required) {
    if (!e) return GSV_ERR_INVALID_ARG;

    /* build refs array */
    static char refs_json[4096];
    refs_json[0] = 0;
    int off = 0;
    for (int i = 0; i < e->voice.n_refs && off < (int)sizeof(refs_json) - 32; i++) {
        off += snprintf(refs_json + off, sizeof(refs_json) - (size_t)off,
                        "%s\"%s\"", i ? "," : "", e->voice.refs[i].name);
    }

    /* report the device a loaded model ACTUALLY ran on (truthful across a
     * CUDA->CPU fallback); before any load, resolve from the config request. */
    char device[16];
    if (e->has_base && e->active_device[0])
        snprintf(device, sizeof(device), "%s", e->active_device);
    else
        gsv_resolve_device(e, device, sizeof(device));

    int len = snprintf(NULL, 0,
        "{\"version\":\"" GSV_VERSION_STR "\","
        "\"device\":\"%s\","
        "\"sample_rate\":%d,"
        "\"loaded_voice\":%s%s%s,"
        "\"refs\":[%s],"
        "\"model_dir\":\"%s\"}",
        device,
        e->has_base ? e->mf.sample_rate : 32000,
        e->has_voice ? "\"" : "",
        e->has_voice ? e->voice.name : "",
        e->has_voice ? "\"" : "null",
        refs_json,
        e->has_base ? e->pkg_dir : "");
    if (required) *required = len + 1;
    if (!buf || buf_size < len + 1) return GSV_OK;
    snprintf(buf, (size_t)len + 1,
        "{\"version\":\"" GSV_VERSION_STR "\","
        "\"device\":\"%s\","
        "\"sample_rate\":%d,"
        "\"loaded_voice\":%s%s%s,"
        "\"refs\":[%s],"
        "\"model_dir\":\"%s\"}",
        device,
        e->has_base ? e->mf.sample_rate : 32000,
        e->has_voice ? "\"" : "",
        e->has_voice ? e->voice.name : "",
        e->has_voice ? "\"" : "null",
        refs_json,
        e->has_base ? e->pkg_dir : "");
    return GSV_OK;
}

/* List the ORT execution providers actually available in this process
 * (comma-joined), e.g. "CPUExecutionProvider,CUDAExecutionProvider". No model is
 * loaded, so it needs no VRAM beyond a light provider init — useful to confirm the
 * CUDA provider DLL + its runtime deps are correctly staged before a full run. */
GSVTTS_API int gsv_available_providers(const gsv_engine *e, char *buf, int buf_size) {
    if (!e || !e->api) return GSV_ERR_INVALID_ARG;
    if (buf && buf_size > 0) buf[0] = '\0';
    char **providers = NULL;
    int count = 0;
    OrtStatus *st = e->api->GetAvailableProviders(&providers, &count);
    if (st) { e->api->ReleaseStatus(st); return GSV_ERR_BACKEND; }
    int off = 0;
    for (int i = 0; i < count && buf && buf_size > 1; i++) {
        const char *p = providers[i] ? providers[i] : "?";
        off += snprintf(buf + off, (size_t)(buf_size - off), "%s%s", i ? "," : "", p);
    }
    e->api->ReleaseAvailableProviders(providers, count);
    return GSV_OK;
}

static const gsv_ref *find_ref(gsv_engine *e, const char *ref_name) {
    for (int i = 0; i < e->voice.n_refs; i++)
        if (strcmp(e->voice.refs[i].name, ref_name) == 0)
            return &e->voice.refs[i];
    return NULL;
}

static int16_t f32_to_s16(float x) {
    if (x > 1.0f) x = 1.0f;
    else if (x < -1.0f) x = -1.0f;
    return (int16_t)(x * 32767.0f);
}

/* ---- sentence segmentation + per-sentence synthesis (M7 / long text) ---- */

/* UTF-8 sequence length (1..4) for a lead byte; 1 for ASCII/continuation/invalid. */
static int utf8_seq_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* Byte length of a terminal-punctuation sequence starting at p, else 0.
 * CJK 。！？； (3-byte) and ASCII . ! ? ; \n (1-byte). */
static int gsv_terminal_punct_len(const unsigned char *p) {
    if (!p || !*p) return 0;
    if (p[0] == 0xE3 && p[1] == 0x80 && p[2] == 0x82) return 3;                 /* 。 */
    if (p[0] == 0xEF && p[1] == 0xBC &&
       (p[2] == 0x81 || p[2] == 0x9F || p[2] == 0x9B)) return 3;                /* ！ ？ ； */
    switch (*p) { case '.': case '!': case '?': case ';': case '\n': return 1; }
    return 0;
}

/* Byte length of a clause-delimiter sequence starting at p (comma-level breaks),
 * else 0. CJK ，：、・ (3-byte) and ASCII , : ~ (1-byte). Used by the chunker;
 * sentence-level breaks are gsv_terminal_punct_len. NOTE: ellipsis (…) is
 * deliberately NOT a cut point — the t2s model renders a natural long pause for
 * the … token itself, and adding real silence on top double-stacks (~1s). */
static int gsv_clause_punct_len(const unsigned char *p) {
    if (!p || !*p) return 0;
    if (p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x8C) return 3;                 /* ， */
    if (p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x9A) return 3;                 /* ： */
    if (p[0] == 0xE3 && p[1] == 0x80 && p[2] == 0x81) return 3;                 /* 、 */
    if (p[0] == 0xE3 && p[1] == 0x83 && p[2] == 0xBB) return 3;                 /* ・ */
    switch (*p) { case ',': case ':': case '~': return 1; }
    return 0;
}

/* 语义长度（与参考实现一致）:中日文字计 1,ASCII 字母/数字连续串计 1.75,其余 0。
 * 用于子句合并阈值(cut_minlen)。 */
static double gsv_semantic_len(const unsigned char *s, int n) {
    double v = 0;
    int alnum_run = 0;
    for (int i = 0; i < n;) {
        unsigned char c = s[i];
        if (c < 0x80) {
            int alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
            if (alnum && !alnum_run) v += 1.75;
            alnum_run = alnum;
            i++;
            continue;
        }
        int w = utf8_seq_len(c);
        if (i + w > n) break;
        uint32_t cp = (uint32_t)(c & (0xFF >> (w + 1)));
        for (int k = 1; k < w; k++) cp = (cp << 6) | (uint32_t)(s[i + k] & 0x3F);
        if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
            (cp >= 0xFF66 && cp <= 0xFF9F)) v += 1.0;
        alnum_run = 0;
        i += w;
    }
    return v;
}

/* 块尾标点 → 静音缩放(cut_mute_scale_map):
 * 。 . ! ? ！ ？ ； ; \n … 1.5;、 ・ 0.8;其余(含无标点的硬切尾)1.0。
 * … 不再作为切点,此处缩放只在整句以省略号收尾时生效。 */
static float gsv_chunk_tail_scale(const char *chunk) {
    if (!chunk || !*chunk) return 1.0f;
    const unsigned char *p = (const unsigned char *)chunk;
    while (*p) { int w = utf8_seq_len(*p); if (!p[w]) break; p += w; }
    if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0xA6) return 1.5f;              /* … */
    if (p[0] == 0xE3 && p[1] == 0x80 && p[2] == 0x82) return 1.5f;              /* 。 */
    if (p[0] == 0xE3 && p[1] == 0x80 && p[2] == 0x81) return 0.8f;              /* 、 */
    if (p[0] == 0xE3 && p[1] == 0x83 && p[2] == 0xBB) return 0.8f;              /* ・ */
    if (p[0] == 0xEF && p[1] == 0xBC &&
       (p[2] == 0x81 || p[2] == 0x9F || p[2] == 0x9B)) return 1.5f;             /* ！ ？ ； */
    switch (*p) { case '.': case '!': case '?': case ';': case '\n': return 1.5f; }
    return 1.0f;
}

/* 文本末尾是否为停顿标点(终止级或子句级) */
static int gsv_ends_with_pause(const char *text) {
    if (!text || !*text) return 0;
    const unsigned char *p = (const unsigned char *)text;
    while (*p) { int w = utf8_seq_len(*p); if (!p[w]) break; p += w; }
    return gsv_terminal_punct_len(p) > 0 || gsv_clause_punct_len(p) > 0;
}

/* 是否含「非标点、非空白」的字符(纯符号/空白文本返回 0) */
static int gsv_has_speakable(const char *text) {
    for (const unsigned char *p = (const unsigned char *)text; *p;) {
        int d = gsv_terminal_punct_len(p);
        if (d == 0) d = gsv_clause_punct_len(p);
        if (d > 0) { p += d; continue; }
        if (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') { p++; continue; }
        return 1;
    }
    return 0;
}

/* Cut UTF-8 text into synthesis chunks (reference-implementation cut_text semantics):
 * split at sentence AND clause punctuation (delimiter kept with its chunk), then
 * merge clauses until the chunk reaches cut_minlen semantic chars; any chunk over
 * max_bytes is hard-split at a char boundary. Returns malloc'd NUL-terminated
 * strings in *out (array of *out_n); caller frees via gsv_free_sentences.
 * GSV_OK or GSV_ERR_NO_MEMORY. */
static int gsv_cut_text(const char *text, int max_bytes, double cut_minlen,
                        char ***out, int *out_n) {
    *out = NULL; *out_n = 0;
    if (!text || !*text) return GSV_OK;
    if (max_bytes < 8) max_bytes = 8;
    int cap = 8, n = 0;
    char **arr = (char **)malloc(sizeof(char *) * (size_t)cap);
    if (!arr) return GSV_ERR_NO_MEMORY;

    char *cur = NULL;               /* current accumulating chunk (NUL-terminated) */
    int cur_len = 0, cur_cap = 0;
    double cur_sem = 0;

    const unsigned char *s = (const unsigned char *)text;
    while (*s) {
        while (*s) {   /* skip leading delimiters / whitespace */
            int dlen = gsv_terminal_punct_len(s);
            if (dlen == 0) dlen = gsv_clause_punct_len(s);
            if (dlen > 0) { s += dlen; continue; }
            if (*s == ' ' || *s == '\t' || *s == '\r') { s++; continue; }
            break;
        }
        if (!*s) break;

        for (;;) {   /* accumulate clauses until minlen reached / text ends / chunk full */
            const unsigned char *start = s;
            int len = 0, delim = 0;
            while (*s) {
                int dlen = gsv_terminal_punct_len(s);
                if (dlen == 0) dlen = gsv_clause_punct_len(s);
                if (dlen > 0) { len += dlen; s += dlen; delim = 1; break; }
                int w = utf8_seq_len(*s);
                if (cur_len + len + w > max_bytes) break;   /* chunk full: hard-split */
                len += w; s += w;
            }
            if (len <= 0) break;                            /* safety: avoid stall */
            if (cur_len + len + 1 > cur_cap) {
                int nc = cur_cap ? cur_cap : 64;
                while (cur_len + len + 1 > nc) nc *= 2;
                char *nb = (char *)realloc(cur, (size_t)nc);
                if (!nb) {
                    free(cur); cur = NULL;
                    for (int i = 0; i < n; i++) free(arr[i]);
                    free(arr);
                    return GSV_ERR_NO_MEMORY;
                }
                cur = nb; cur_cap = nc;
            }
            memcpy(cur + cur_len, start, (size_t)len);
            cur_len += len;
            cur[cur_len] = 0;
            cur_sem += gsv_semantic_len(start, len);
            if (!delim) break;                    /* text end or hard split -> flush */
            if (cur_sem >= cut_minlen) break;     /* threshold reached -> flush */
        }
        if (cur_len > 0) {
            if (n == cap) {
                cap *= 2;
                char **na = (char **)realloc(arr, sizeof(char *) * (size_t)cap);
                if (!na) {
                    free(cur); cur = NULL;
                    for (int i = 0; i < n; i++) free(arr[i]);
                    free(arr);
                    return GSV_ERR_NO_MEMORY;
                }
                arr = na;
            }
            arr[n++] = cur;
            cur = NULL; cur_len = cur_cap = 0; cur_sem = 0;
        }
    }
    if (cur_len > 0) {
        if (n == cap) {
            cap *= 2;
            char **na = (char **)realloc(arr, sizeof(char *) * (size_t)cap);
            if (!na) { free(cur); for (int i = 0; i < n; i++) free(arr[i]); free(arr); return GSV_ERR_NO_MEMORY; }
            arr = na;
        }
        arr[n++] = cur;
    } else {
        free(cur);
    }
    *out = arr; *out_n = n;
    return GSV_OK;
}

static void gsv_free_sentences(char **sents, int n) {
    if (sents) { for (int i = 0; i < n; i++) free(sents[i]); free(sents); }
}

/* t2s + SoVITS decode for a pre-G2Ped target: phones2/n2 (+ zh bert2) -> float audio.
 * The faithful single-shot path shared by gsv_infer and the per-sentence helpers. */
static int gsv_t2s_decode(gsv_engine *e, const gsv_ref *ref, const int64_t *phones2, int n2,
                          const float *bert2, float speed, float **out_audio, int *out_len) {
    *out_audio = NULL; *out_len = 0;
    int n1 = ref->n_phones1;
    int lx = n1 + n2;
    int ly = ref->n_prompt_codes;
    int max_len = e->mf.t2s.max_len;

    int64_t *x_ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)(lx > 0 ? lx : 1));
    int64_t *y64   = (int64_t *)malloc(sizeof(int64_t) * (size_t)ly);
    if (!x_ids || !y64) { free(x_ids); free(y64); return GSV_ERR_NO_MEMORY; }
    for (int i = 0; i < n1; i++) x_ids[i] = ref->phones1 ? ref->phones1[i] : 0;
    for (int i = 0; i < n2; i++) x_ids[n1 + i] = phones2[i];
    for (int i = 0; i < ly; i++) y64[i] = (int64_t)ref->prompt_codes[i];

    /* Reference always builds bert = cat([bert1, bert2]); each segment holds real
     * features or zeros depending on ITS OWN language. Build the full [lx*1024]
     * array (zero base) and fill per-segment: prompt from ref->bert (else zeros),
     * target from bert2 (else zeros). The old code only built it when BOTH were
     * present, so an en/ja target with a Chinese prompt got all-zeros BERT -> the
     * GPT lost its prompt conditioning and hit EOS after ~9 tokens (near-silent). */
    float *bert = NULL; int own_bert = 0;
    if ((n1 > 0 && ref->bert) || (n2 > 0 && bert2)) {
        bert = (float *)calloc((size_t)lx * 1024, sizeof(float));   /* zero base */
        if (bert) {
            if (n1 > 0 && ref->bert)
                memcpy(bert, ref->bert, (size_t)n1 * 1024 * sizeof(float));
            if (n2 > 0 && bert2)
                memcpy(bert + (size_t)n1 * 1024, bert2, (size_t)n2 * 1024 * sizeof(float));
            own_bert = 1;
        }
    }

    int *codes = (int *)malloc(sizeof(int) * (size_t)(max_len > 0 ? max_len : 1));
    if (!codes) { if (own_bert) free(bert); free(x_ids); free(y64); return GSV_ERR_NO_MEMORY; }
    int n_codes = 0;
    int rc = gsv_t2s_generate(e, x_ids, lx, y64, ly, bert, codes, max_len, &n_codes);
    free(x_ids); free(y64); if (own_bert) free(bert);
    if (rc != GSV_OK) { free(codes); return rc; }
    if (n_codes <= 0) { free(codes); gsv_set_error(e, "t2s produced no codes"); return GSV_ERR_TEXT_EMPTY; }

    int spf = e->voice.cfg.samples_per_frame > 0 ? e->voice.cfg.samples_per_frame : 640;
    float spd = speed < 0.1f ? 0.1f : speed;
    int acap = (int)((2.0 * (double)n_codes / spd) * (double)spf) + 8192;
    float *audio = NULL;
    int alen = 0;
    for (int attempt = 0; attempt < 8; attempt++) {
        float *buf = (float *)realloc(audio, sizeof(float) * (size_t)acap);
        if (!buf) { free(audio); free(codes); return GSV_ERR_NO_MEMORY; }
        audio = buf;
        rc = gsv_sovits_decode(e, ref->name, codes, n_codes, phones2, n2,
                               e->sampling.noise_scale, speed, audio, acap, &alen);
        if (rc == GSV_OK) break;
        if (rc != GSV_ERR_NO_MEMORY) { free(audio); free(codes); return rc; }
        acap *= 2;
    }
    free(codes);
    if (rc != GSV_OK) { free(audio); return rc; }
    *out_audio = audio; *out_len = alen;
    return GSV_OK;
}

/* G2P one sentence + t2s/decode -> float audio. Empty / non-speakable sentence
 * yields 0 samples (GSV_OK). */
static int gsv_synth_sentence(gsv_engine *e, const gsv_ref *ref, const char *sent,
                              const char *lang, float speed, float **out_audio, int *out_len) {
    *out_audio = NULL; *out_len = 0;
    if (!sent || !*sent) return GSV_OK;
    gsv_g2p_full_result fr;
    int rc = gsv_g2p_dispatch_full(e->g2p, sent, lang, &fr);
    if (rc != GSV_OK) { gsv_set_error(e, "g2p failed for sentence (%d)", rc); return rc; }
    int64_t *phones2 = fr.ids; int n2 = fr.n_ids; fr.ids = NULL;
    if (n2 <= 0) { gsv_g2p_full_result_free(&fr); return GSV_OK; }
    float *bert2 = NULL;
    if (!strcmp(lang, "zh") && e->bert_tok.n && fr.n_words == fr.n_phs && fr.n_words > 0) {
        bert2 = (float *)malloc(sizeof(float) * (size_t)n2 * 1024);
        if (bert2 && gsv_bert_phone_features(e, fr.words, fr.n_words, fr.phs, n2, bert2) != GSV_OK) { free(bert2); bert2 = NULL; }
    }
    gsv_g2p_full_result_free(&fr);
    rc = gsv_t2s_decode(e, ref, phones2, n2, bert2, speed, out_audio, out_len);
    free(phones2); free(bert2);
    return rc;
}

/* Chunked synthesis: each chunk runs its own G2P + t2s + decode; a ~10ms fade-in
 * at each chunk start avoids clicks at the joins, real silence of
 * cut_mute / speed × tail-scale is appended after every chunk (… 2.0, sentence
 * puncts 1.5, 、・ 0.8, else 1.0), and a fixed 0.2s tail ends the whole output
 * (reference-implementation style punctuation pauses). */
static int gsv_infer_chunks(gsv_engine *e, const gsv_ref *ref, const char *text,
                            const char *lang, float speed,
                            int16_t *out_pcm, int64_t out_capacity, int64_t *written_samples) {
    int budget = e->mf.t2s.max_len - ref->n_phones1 - ref->n_prompt_codes;
    if (budget < 8) { gsv_set_error(e, "gsv_infer: prompt too large vs max_len %d", e->mf.t2s.max_len); return GSV_ERR_INVALID_ARG; }
    char **chunks = NULL; int nchunk = 0;
    int rc = gsv_cut_text(text, budget, (double)e->cut_minlen, &chunks, &nchunk);
    if (rc != GSV_OK) return rc;
    if (nchunk == 0) { gsv_free_sentences(chunks, nchunk); gsv_set_error(e, "gsv_infer: no speakable text"); return GSV_ERR_TEXT_EMPTY; }

    float mute = e->cut_mute / (speed < 0.1f ? 0.1f : speed);
    int rate = e->mf.sample_rate;
    int cap = 8192, total = 0;
    float *acc = (float *)malloc(sizeof(float) * (size_t)cap);
    if (!acc) { gsv_free_sentences(chunks, nchunk); return GSV_ERR_NO_MEMORY; }

    for (int i = 0; i < nchunk; i++) {
        float *sa = NULL; int slen = 0;
        rc = gsv_synth_sentence(e, ref, chunks[i], lang, speed, &sa, &slen);
        if (rc != GSV_OK) { free(sa); free(acc); gsv_free_sentences(chunks, nchunk); return rc; }
        int pause_here = (slen > 0) ? (int)(mute * gsv_chunk_tail_scale(chunks[i]) * rate + 0.5f) : 0;
        if (slen > 0) {
            int fn = slen < 320 ? slen : 320;   /* ~10ms @32k fade-in against join clicks */
            for (int k = 0; k < fn; k++) sa[k] *= (float)k / (float)fn;
            while (total + slen + pause_here > cap) {
                cap *= 2;
                float *na = (float *)realloc(acc, sizeof(float) * (size_t)cap);
                if (!na) { free(sa); free(acc); gsv_free_sentences(chunks, nchunk); return GSV_ERR_NO_MEMORY; }
                acc = na;
            }
            memcpy(acc + total, sa, (size_t)slen * sizeof(float));
            total += slen;
            if (pause_here) { memset(acc + total, 0, (size_t)pause_here * sizeof(float)); total += pause_here; }
        }
        free(sa);
    }
    gsv_free_sentences(chunks, nchunk);

    int trail = (int)(0.2f * rate + 0.5f);   /* fixed 0.2s tail (same as reference) */
    while (total + trail > cap) {
        cap *= 2;
        float *na = (float *)realloc(acc, sizeof(float) * (size_t)cap);
        if (!na) { free(acc); return GSV_ERR_NO_MEMORY; }
        acc = na;
    }
    memset(acc + total, 0, (size_t)trail * sizeof(float));
    total += trail;

    int64_t nwrite = total < out_capacity ? total : out_capacity;
    for (int64_t i = 0; i < nwrite; i++) out_pcm[i] = f32_to_s16(acc[i]);
    *written_samples = total;
    free(acc);
    return GSV_OK;
}

static int gsv_infer_impl(gsv_engine *e, const char *text,
              const char *lang, float speed, const char *ref_name,
              int16_t *out_pcm, int64_t out_capacity, int64_t *written_samples) {
    if (!e || !e->has_voice) return GSV_ERR_NOT_LOADED;
    if (!text || !text[0]) { gsv_set_error(e, "gsv_infer: empty text"); return GSV_ERR_TEXT_EMPTY; }
    if (!ref_name || !ref_name[0]) { gsv_set_error(e, "gsv_infer: bad ref_name"); return GSV_ERR_INVALID_ARG; }
    if (speed <= 0.0f) { gsv_set_error(e, "gsv_infer: speed must be > 0"); return GSV_ERR_INVALID_ARG; }
    if (!written_samples) return GSV_ERR_INVALID_ARG;
    *written_samples = 0;
    if (out_capacity < 0 || (out_capacity > 0 && !out_pcm)) return GSV_ERR_INVALID_ARG;

    const gsv_ref *ref = find_ref(e, ref_name);
    if (!ref) { gsv_set_error(e, "gsv_infer: ref '%s' not warmed up", ref_name); return GSV_ERR_REF_MISSING; }
    if (!ref->prompt_codes || ref->n_prompt_codes <= 0) {
        gsv_set_error(e, "gsv_infer: ref '%s' has no prompt codes", ref_name);
        return GSV_ERR_REF_MISSING;
    }

    /* target language: explicit arg > ref's lang > zh */
    const char *tlang = (lang && lang[0]) ? lang : (ref->lang[0] ? ref->lang : "zh");
    if (!e->g2p || !gsv_g2p_dispatch_has(e->g2p, tlang)) {
        gsv_set_error(e, "gsv_infer: G2P unavailable for lang '%s'", tlang);
        return GSV_ERR_NOT_LOADED;
    }

    /* Reference _check_pause semantics: text not ending in pause punctuation gets a "." appended
     * so the final chunk also carries its tail pause. Pure-symbol input still yields
     * no chunks -> TEXT_EMPTY below. */
    const char *text2 = text;
    char *dotbuf = NULL;
    if (!gsv_ends_with_pause(text) && gsv_has_speakable(text)) {
        size_t tl = strlen(text);
        dotbuf = (char *)malloc(tl + 2);
        if (dotbuf) {
            memcpy(dotbuf, text, tl);
            dotbuf[tl] = '.'; dotbuf[tl + 1] = 0;
            text2 = dotbuf;
        }
    }
    int rc = gsv_infer_chunks(e, ref, text2, tlang, speed,
                              out_pcm, out_capacity, written_samples);
    free(dotbuf);
    return rc;
}

int gsv_infer(gsv_engine *e, const char *text,
              const char *lang, float speed, const char *ref_name,
              int16_t *out_pcm, int64_t out_capacity, int64_t *written_samples) {
    if (!e) return GSV_ERR_NOT_LOADED;
    pthread_mutex_lock(&e->infer_lock);
    int rc = gsv_infer_impl(e, text, lang, speed, ref_name,
                            out_pcm, out_capacity, written_samples);
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

/* M6b verification helper: run G2P + per-phone BERT features for `text` and write
 * a binary dump (little-endian) to out_path:
 *   int32 n_phones, int32 n_words,
 *   [int32 len + len bytes] per word (UTF-8),
 *   int32 phs[n_words],
 *   int32 have_feats (0/1),
 *   float32 feats[n_phones*1024]  (only if have_feats) */
static int gsv_bert_dump_file_impl(gsv_engine *e, const char *text, const char *lang,
                                   const char *out_path) {
    if (!e || !e->g2p || !text || !out_path) return GSV_ERR_INVALID_ARG;
    gsv_g2p_full_result fr;
    int rc = gsv_g2p_dispatch_full(e->g2p, text, lang, &fr);
    if (rc != GSV_OK) return rc;
    FILE *f = fopen(out_path, "wb");
    if (!f) { gsv_g2p_full_result_free(&fr); return GSV_ERR_BACKEND; }
    int n_phones = fr.n_ids, n_words = fr.n_words;
    fwrite(&n_phones, 4, 1, f);
    fwrite(&n_words, 4, 1, f);
    for (int i = 0; i < n_words; i++) { int L = (int)strlen(fr.words[i]); fwrite(&L, 4, 1, f); if (L) fwrite(fr.words[i], 1, (size_t)L, f); }
    for (int i = 0; i < n_words; i++) { int v = (fr.phs && i < fr.n_phs) ? fr.phs[i] : 0; fwrite(&v, 4, 1, f); }
    float *feats = NULL; int have = 0, prc = GSV_ERR_NOT_LOADED;
    if (e->bert_tok.n && fr.n_words == fr.n_phs && n_phones > 0) {
        feats = (float *)malloc(sizeof(float) * (size_t)n_phones * 1024);
        if (feats) { prc = gsv_bert_phone_features(e, fr.words, fr.n_words, fr.phs, n_phones, feats); if (prc == GSV_OK) have = 1; }
    }
    fwrite(&have, 4, 1, f);
    fwrite(&prc, 4, 1, f);
    if (have) fwrite(feats, 4, (size_t)n_phones * 1024, f);
    fclose(f);
    free(feats);
    gsv_g2p_full_result_free(&fr);
    return GSV_OK;
}

GSVTTS_API int gsv_bert_dump_file(gsv_engine *e, const char *text, const char *lang,
                                    const char *out_path) {
    if (!e) return GSV_ERR_INVALID_ARG;
    pthread_mutex_lock(&e->infer_lock);
    int rc = gsv_bert_dump_file_impl(e, text, lang, out_path);
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

/* Expose the language G2P as a public query: text -> final phoneme id sequence
 * (same path gsv_infer uses). Additive API for callers wanting phonemes without
 * running full synthesis, and for G2P cross-checks. */
GSVTTS_API int gsv_text_to_phoneme_ids(gsv_engine *e, const char *text, const char *lang,
                                        int64_t *out_ids, int cap, int *out_n) {
    if (!e || !e->g2p) return GSV_ERR_NOT_LOADED;
    pthread_mutex_lock(&e->infer_lock);
    int rc = gsv_g2p_dispatch_ids(e->g2p, text, lang, out_ids, cap, out_n);
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

/* Pull-based streaming state: text pre-segmented into sentences; each next()
 * decodes the current sentence on demand and drains it in <=out_capacity chunks. */
struct gsv_stream {
    gsv_engine *e;
    char ref_name[128];
    char lang[8];
    float speed;
    char **sents;   int n_sent; int cur;        /* sentence list + current index */
    float *cur_audio; int cur_len; int cur_pos;  /* decoded audio for `cur` + read offset */
};

gsv_stream *gsv_stream_begin(gsv_engine *e,
                              const char *text, const char *lang, float speed,
                              const char *ref_name) {
    if (!e || !e->has_voice) return NULL;
    if (!text || !text[0]) { gsv_set_error(e, "gsv_stream_begin: empty text"); return NULL; }
    if (!ref_name || !ref_name[0]) { gsv_set_error(e, "gsv_stream_begin: bad ref_name"); return NULL; }
    if (speed <= 0.0f) { gsv_set_error(e, "gsv_stream_begin: speed must be > 0"); return NULL; }

    pthread_mutex_lock(&e->infer_lock);
    const gsv_ref *ref = find_ref(e, ref_name);
    if (!ref || !ref->prompt_codes || ref->n_prompt_codes <= 0) {
        gsv_set_error(e, "gsv_stream_begin: ref '%s' not warmed up / no prompt codes", ref_name);
        pthread_mutex_unlock(&e->infer_lock); return NULL;
    }
    const char *tlang = (lang && lang[0]) ? lang : (ref->lang[0] ? ref->lang : "zh");
    if (!e->g2p || !gsv_g2p_dispatch_has(e->g2p, tlang)) {
        gsv_set_error(e, "gsv_stream_begin: G2P unavailable for lang '%s'", tlang);
        pthread_mutex_unlock(&e->infer_lock); return NULL;
    }
    int budget = e->mf.t2s.max_len - ref->n_phones1 - ref->n_prompt_codes;
    if (budget < 8) {
        gsv_set_error(e, "gsv_stream_begin: prompt too large vs max_len %d", e->mf.t2s.max_len);
        pthread_mutex_unlock(&e->infer_lock); return NULL;
    }

    gsv_stream *s = (gsv_stream *)calloc(1, sizeof(gsv_stream));
    if (!s) { gsv_set_error(e, "gsv_stream_begin: OOM"); pthread_mutex_unlock(&e->infer_lock); return NULL; }
    s->e = e;
    snprintf(s->ref_name, sizeof(s->ref_name), "%s", ref_name);
    snprintf(s->lang, sizeof(s->lang), "%s", tlang);
    s->speed = speed;
    int rc = gsv_cut_text(text, budget, (double)e->cut_minlen, &s->sents, &s->n_sent);
    if (rc != GSV_OK || s->n_sent == 0) {
        gsv_set_error(e, "gsv_stream_begin: no speakable text");
        gsv_free_sentences(s->sents, s->n_sent); free(s);
        pthread_mutex_unlock(&e->infer_lock); return NULL;
    }
    pthread_mutex_unlock(&e->infer_lock);
    return s;
}

int gsv_stream_next(gsv_stream *s, int16_t *out_pcm, int64_t out_capacity,
                    int64_t *written) {
    if (!s || !written) return GSV_ERR_INVALID_ARG;
    *written = 0;
    if (out_capacity < 0 || (out_capacity > 0 && !out_pcm)) return GSV_ERR_INVALID_ARG;

    pthread_mutex_lock(&s->e->infer_lock);
    const gsv_ref *ref = find_ref(s->e, s->ref_name);
    if (!ref) { pthread_mutex_unlock(&s->e->infer_lock); return GSV_ERR_REF_MISSING; }

    for (;;) {
        if (s->cur >= s->n_sent) { pthread_mutex_unlock(&s->e->infer_lock); return GSV_STREAM_DONE; }
        if (s->cur_audio == NULL) {
            int rc = gsv_synth_sentence(s->e, ref, s->sents[s->cur], s->lang, s->speed,
                                        &s->cur_audio, &s->cur_len);
            if (rc != GSV_OK) { pthread_mutex_unlock(&s->e->infer_lock); return rc; }
            s->cur_pos = 0;
            /* chunk-boundary pause: cut_mute / speed × tail scale, after every non-empty
             * chunk (the last one included — matches the reference is_final behavior) */
            float mute = s->e->cut_mute / (s->speed < 0.1f ? 0.1f : s->speed);
            int psamp = (int)(mute * gsv_chunk_tail_scale(s->sents[s->cur]) * s->e->mf.sample_rate + 0.5f);
            if (psamp > 0 && s->cur_len > 0) {
                float *na = (float *)realloc(s->cur_audio, sizeof(float) * (size_t)(s->cur_len + psamp));
                if (na) {
                    memset(na + s->cur_len, 0, (size_t)psamp * sizeof(float));
                    s->cur_audio = na;
                    s->cur_len += psamp;
                }
            }
        }
        int avail = s->cur_len - s->cur_pos;
        if (avail > 0) {
            int64_t n = avail < out_capacity ? avail : out_capacity;
            for (int64_t i = 0; i < n; i++) out_pcm[i] = f32_to_s16(s->cur_audio[s->cur_pos + i]);
            s->cur_pos += (int)n;
            *written = n;
            pthread_mutex_unlock(&s->e->infer_lock); return GSV_OK;
        }
        /* current sentence produced no audio: advance */
        free(s->cur_audio); s->cur_audio = NULL; s->cur_len = 0; s->cur_pos = 0; s->cur++;
    }
}

void gsv_stream_end(gsv_stream *s) {
    if (!s) return;
    gsv_free_sentences(s->sents, s->n_sent);
    free(s->cur_audio);
    free(s);
}
