/* engine.c - model package load/unload, device resolution, RNG, sampling config */
#include "engine.h"
#include "g2p_dispatch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void path_join(char *out, size_t n, const char *dir, const char *name) {
    size_t dl = strlen(dir);
    if (dl && dir[dl - 1] != '\\' && dir[dl - 1] != '/')
        snprintf(out, n, "%s\\%s", dir, name);
    else
        snprintf(out, n, "%s%s", dir, name);
}

/* GPT models are pure-fp16-only (fixed filename, no variant). SoVITS (enc_p/flowdec) ship
 * BOTH dtypes: _fp32 for CPU (ORT CPU EP has sparse fp16 op coverage; flowdec runs ~1.7x
 * slower in fp16 on x86) and _fp16 for CUDA. The variant is chosen by device, overridable
 * via use_fp16 (0=force fp32, 1=force fp16, -1=auto). */
static const char *sovits_variant(const gsv_engine *e, const char *device) {
    if (e->use_fp16 == 0) return "fp32";
    if (e->use_fp16 == 1) return "fp16";
    return (device && strcmp(device, "cuda") == 0) ? "fp16" : "fp32";
}

static int file_exists(const char *p) {
    DWORD a = GetFileAttributesA(p);
    return (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY));
}

/* out = dirname(path): strip the trailing filename component. */
static void derive_dirname(const char *path, char *out, size_t n) {
    snprintf(out, n, "%s", path);
    char *bs = strrchr(out, '\\');
    char *fs = strrchr(out, '/');
    char *last = (bs && (!fs || bs > fs)) ? bs : fs;
    if (last) *last = '\0';
    if (!out[0]) { out[0] = '.'; out[1] = '\0'; }
}

/* Resolve a per-character model file: try <root>/<prefix>/<name>/<file>, else
 * <root>/<fallback>/<file> (the base/底模 dir). Writes the full path to `out`;
 * returns 1 if a file was found, 0 otherwise. */
static int resolve_named(char *out, size_t n, const char *root,
                         const char *prefix, const char *fallback,
                         const char *name, const char *file) {
    if (prefix && prefix[0] && name && name[0]) {
        char d1[512], d2[640], cand[768];
        path_join(d1, sizeof(d1), root, prefix);
        path_join(d2, sizeof(d2), d1, name);
        path_join(cand, sizeof(cand), d2, file);
        if (file_exists(cand)) { snprintf(out, n, "%s", cand); return 1; }
    }
    if (fallback && fallback[0]) {
        char fd[512], fcand[640];
        path_join(fd, sizeof(fd), root, fallback);
        path_join(fcand, sizeof(fcand), fd, file);
        if (file_exists(fcand)) { snprintf(out, n, "%s", fcand); return 1; }
    }
    return 0;
}

/* Which GSV versions this build supports. Short ids are canonical ("v2pp"); the legacy
 * long form "v2ProPlus" is tolerated. v4 is reserved (not yet implemented). */
static int version_supported(const char *v) {
    if (!v || !v[0]) return 0;
    return strcmp(v, "v2pp") == 0 || strcmp(v, "v2ProPlus") == 0;
}

int gsv_resolve_device(const gsv_engine *e, char *out, int n) {
    if (strcmp(e->device, "cpu") == 0 || strcmp(e->device, "cuda") == 0) {
        snprintf(out, n, "%s", e->device);
        return 0;
    }
    /* auto: ask ORT which EPs are actually available */
    char **providers = NULL;
    int count = 0;
    OrtStatus *st = e->api->GetAvailableProviders(&providers, &count);
    if (st) { e->api->ReleaseStatus(st); snprintf(out, n, "cpu"); return 0; }
    int have_cuda = 0;
    for (int i = 0; i < count; i++) {
        if (providers[i] && strcmp(providers[i], "CUDAExecutionProvider") == 0) have_cuda = 1;
    }
    e->api->ReleaseAvailableProviders(providers, count);
    snprintf(out, n, "%s", have_cuda ? "cuda" : "cpu");
    return 0;
}

/* ---------------- RNG: xorshift64* (seedable, reproducible) ---------------- */
uint64_t gsv_rng_next(gsv_engine *e) {
    uint64_t x = e->rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    e->rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

float gsv_rng_uniform(gsv_engine *e) {   /* (0,1) */
    uint64_t u = gsv_rng_next(e) >> 11;  /* top 53 bits */
    float f = (float)u * (1.0f / 9007199254740992.0f);
    if (f <= 0.0f) f = 1e-9f;
    return f;
}

/* Standard normal via Box-Muller (consumes two uniform draws). */
float gsv_rng_gaussian(gsv_engine *e) {
    float u1 = gsv_rng_uniform(e);
    float u2 = gsv_rng_uniform(e);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

/* ---------------- sampling config ---------------- */
void gsv_sampling_defaults(gsv_sampling *s) {
    s->top_k = 15;
    s->top_p = 1.0f;
    s->temperature = 1.0f;
    s->repetition_penalty = 1.35f;
    s->noise_scale = 0.5f;
}

int gsv_set_sampling(gsv_engine *e, const gsv_sampling *s) {
    if (!e) return GSV_ERR_INVALID_ARG;
    if (!s) { gsv_set_error(e, "gsv_set_sampling: NULL arg"); return GSV_ERR_INVALID_ARG; }
    /* top_k: 0 = disabled, 1..N = keep top-N */
    if (s->top_k < 0 || s->top_p < 0.0f || s->top_p > 1.0f || s->temperature <= 0.0f) {
        gsv_set_error(e, "gsv_set_sampling: invalid range");
        return GSV_ERR_INVALID_ARG;
    }
    pthread_mutex_lock(&e->infer_lock);
    e->sampling = *s;
    pthread_mutex_unlock(&e->infer_lock);
    return GSV_OK;
}

int gsv_get_sampling(const gsv_engine *e, gsv_sampling *s) {
    if (!e || !s) return GSV_ERR_INVALID_ARG;
    *s = e->sampling;
    return GSV_OK;
}

int gsv_set_sentence_pause(gsv_engine *e, float seconds) {
    if (!e) return GSV_ERR_INVALID_ARG;
    if (seconds < 0.0f) { gsv_set_error(e, "gsv_set_sentence_pause: must be >= 0"); return GSV_ERR_INVALID_ARG; }
    pthread_mutex_lock(&e->infer_lock);
    e->sentence_pause = seconds;
    pthread_mutex_unlock(&e->infer_lock);
    return GSV_OK;
}

/* Chunked-synthesis pause controls (reference-implementation style). cut_mute: real silence
 * appended after every synthesis chunk (seconds, divided by speed, scaled by the
 * chunk's tail punctuation). cut_minlen: clauses are merged until the chunk reaches
 * this semantic length (CJK char = 1, ASCII alnum run = 1.75); 0 (default) cuts at
 * every punctuation mark, a huge value disables cutting. */
int gsv_set_cut_mute(gsv_engine *e, float seconds) {
    if (!e) return GSV_ERR_INVALID_ARG;
    if (seconds < 0.0f) { gsv_set_error(e, "gsv_set_cut_mute: must be >= 0"); return GSV_ERR_INVALID_ARG; }
    pthread_mutex_lock(&e->infer_lock);
    e->cut_mute = seconds;
    pthread_mutex_unlock(&e->infer_lock);
    return GSV_OK;
}

int gsv_set_cut_minlen(gsv_engine *e, float chars) {
    if (!e) return GSV_ERR_INVALID_ARG;
    if (chars < 0.0f) { gsv_set_error(e, "gsv_set_cut_minlen: must be >= 0"); return GSV_ERR_INVALID_ARG; }
    pthread_mutex_lock(&e->infer_lock);
    e->cut_minlen = chars;
    pthread_mutex_unlock(&e->infer_lock);
    return GSV_OK;
}

/* fp16 model-variant preference. mode: -1 auto (cuda prefers *_fp16.onnx when a
 * sibling exists, else fp32), 0 off (always fp32), 1 on (prefer *_fp16.onnx).
 * Takes effect at gsv_load_model time. */
int gsv_set_fp16(gsv_engine *e, int mode) {
    if (!e) return GSV_ERR_INVALID_ARG;
    if (mode < -1 || mode > 1) { gsv_set_error(e, "gsv_set_fp16: mode must be -1|0|1"); return GSV_ERR_INVALID_ARG; }
    pthread_mutex_lock(&e->infer_lock);
    e->use_fp16 = mode;
    pthread_mutex_unlock(&e->infer_lock);
    return GSV_OK;
}

int gsv_get_sentence_pause(const gsv_engine *e, float *seconds) {
    if (!e || !seconds) return GSV_ERR_INVALID_ARG;
    *seconds = e->sentence_pause;
    return GSV_OK;
}

int gsv_get_cut_mute(const gsv_engine *e, float *seconds) {
    if (!e || !seconds) return GSV_ERR_INVALID_ARG;
    *seconds = e->cut_mute;
    return GSV_OK;
}

int gsv_get_cut_minlen(const gsv_engine *e, float *chars) {
    if (!e || !chars) return GSV_ERR_INVALID_ARG;
    *chars = e->cut_minlen;
    return GSV_OK;
}

/* ---------------- voice release ---------------- */
static void free_voice(gsv_engine *e) {
    gsv_voice *v = &e->voice;
    for (int i = 0; i < v->n_refs; i++) {
        if (v->refs[i].prompt_codes) free(v->refs[i].prompt_codes);
        if (v->refs[i].ge) free(v->refs[i].ge);
        if (v->refs[i].ge512) free(v->refs[i].ge512);
    }
    free(v->refs);
    v->refs = NULL; v->n_refs = 0;
    gsv_sess_destroy(e->api, &v->enc_p);
    gsv_sess_destroy(e->api, &v->flowdec);
    memset(v, 0, sizeof(*v));
}

/* per-character tier: GPT sessions + SoVITS sessions + cached refs. Freed on each
 * gsv_load_model (single-resident switch) and on gsv_unload_model. Base is untouched. */
static void free_character(gsv_engine *e) {
    gsv_sess_destroy(e->api, &e->t2s_prefill);
    gsv_sess_destroy(e->api, &e->t2s_step);
    free_voice(e);
}

/* base tier: feature models + VQ + G2P + BERT vocab + t2s scratch. Freed only at engine
 * destroy (persists across character loads). */
static void free_base(gsv_engine *e) {
    gsv_sess_destroy(e->api, &e->bert);
    gsv_sess_destroy(e->api, &e->hubert);
    gsv_sess_destroy(e->api, &e->sv);
    gsv_sess_destroy(e->api, &e->ge);
    free(e->codebook); e->codebook = NULL;
    free(e->ssl_proj_w); e->ssl_proj_w = NULL;
    free(e->ssl_proj_b); e->ssl_proj_b = NULL;
    /* t2s scratch (M4): sized by version params, persists */
    free(e->t2s_kbuf); e->t2s_kbuf = NULL;
    free(e->t2s_vbuf); e->t2s_vbuf = NULL;
    free(e->t2s_idx);  e->t2s_idx = NULL;
    /* G2P dispatcher (M6) */
    if (e->g2p) { gsv_g2p_dispatch_destroy(e->g2p); e->g2p = NULL; }
    /* BERT vocab (M6b) */
    gsv_bert_tok_free(&e->bert_tok);
}

static int read_f32_file(const char *path, float **out, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t need = n * sizeof(float);
    float *buf = (float *)malloc(need);
    if (!buf || fread(buf, 1, need, f) != need) { free(buf); fclose(f); return -1; }
    fclose(f);
    *out = buf;
    return 0;
}

/* Lazily create the zh-only bert.onnx session on first use (P2 memory: en/ja never
 * pay for the ~1.2 GB fp32 graph). Must be called under e->infer_lock. Non-fatal:
 * on failure set bert_failed and return -1 so callers degrade BERT to zeros. */
int gsv_bert_ensure(gsv_engine *e) {
    if (e->bert_loaded)  return 0;
    if (e->bert_failed)  return -1;
    if (!e->has_base || !e->mf.bert.onnx[0]) { e->bert_failed = 1; return -1; }
    char p[700];
    path_join(p, sizeof(p), e->pkg_dir, e->mf.bert.onnx);   /* rel path from manifest */
    if (gsv_sess_create(e->api, e->env, p, e->device, e->intra_op_threads, &e->bert) != 0) {
        gsv_set_error(e, "gsv_bert_ensure: ORT session create failed for bert (%s)", p);
        e->bert_failed = 1;
        return -1;
    }
    e->bert_loaded = 1;
    return 0;
}

/* ---------------- base load (init) ---------------- */
/* Load the base-tier sessions + data for a resolved device. Paths are relative to
 * e->pkg_dir (the base manifest's dir). bert.onnx itself stays lazy (gsv_bert_ensure). */
static int load_base_sessions(gsv_engine *e, const char *device) {
    char p[700];
    struct { gsv_sess *s; const char *file; const char *what; } shared[] = {
        { &e->hubert, e->mf.hubert.onnx, "hubert" },
        { &e->sv,     e->mf.sv.onnx,     "sv" },
        { &e->ge,     e->mf.ge.onnx,     "ge" },
    };
    for (size_t i = 0; i < sizeof(shared) / sizeof(shared[0]); i++) {
        if (!shared[i].file[0]) continue;
        path_join(p, sizeof(p), e->pkg_dir, shared[i].file);   /* manifest holds *_fp16 path */
        if (gsv_sess_create(e->api, e->env, p, device, e->intra_op_threads, shared[i].s) != 0) {
            gsv_set_error(e, "load_base: ORT session create failed for %s (%s)", shared[i].what, p);
            return GSV_ERR_BACKEND;
        }
    }

    /* reset lazy BERT state + load the (small) vocab; the big onnx is created on first zh use */
    e->bert_loaded = 0;
    e->bert_failed = 0;
    if (e->mf.bert.vocab[0]) {
        path_join(p, sizeof(p), e->pkg_dir, e->mf.bert.vocab);
        if (gsv_bert_tok_load(&e->bert_tok, p) != 0)
            gsv_bert_tok_free(&e->bert_tok);   /* ensure clean state */
    }

    /* codebook */
    path_join(p, sizeof(p), e->pkg_dir, e->mf.vq.codebook);
    if (read_f32_file(p, &e->codebook,
                      (size_t)e->mf.vq.bins * (size_t)e->mf.vq.dim) != 0) {
        gsv_set_error(e, "load_base: codebook read failed (%s)", p);
        return GSV_ERR_IO;
    }

    /* ssl_proj (prompt-codes Conv1d k=2 s=2): one file, weight [dim,dim,k] then bias [dim] */
    if (e->mf.vq.ssl_proj_file[0]) {
        path_join(p, sizeof(p), e->pkg_dir, e->mf.vq.ssl_proj_file);
        size_t dw = (size_t)e->mf.vq.dim * (size_t)e->mf.vq.dim * (size_t)e->mf.vq.ssl_proj_kernel;
        size_t nb = (size_t)e->mf.vq.dim;
        FILE *f = fopen(p, "rb");
        if (!f) { gsv_set_error(e, "load_base: ssl_proj read failed (%s)", p); return GSV_ERR_IO; }
        e->ssl_proj_w = (float *)malloc(dw * sizeof(float));
        e->ssl_proj_b = (float *)malloc(nb * sizeof(float));
        if (!e->ssl_proj_w || !e->ssl_proj_b ||
            fread(e->ssl_proj_w, 1, dw * sizeof(float), f) != dw * sizeof(float) ||
            fread(e->ssl_proj_b, 1, nb * sizeof(float), f) != nb * sizeof(float)) {
            gsv_set_error(e, "load_base: ssl_proj read failed (%s)", p);
            fclose(f); return GSV_ERR_IO;
        }
        fclose(f);
    }

    /* G2P dispatcher (M6): language data from <pkg>/g2p. Missing sub-dirs are tolerated. */
    char g2pbase[700];
    path_join(g2pbase, sizeof(g2pbase), e->pkg_dir, "g2p");
    e->g2p = gsv_g2p_dispatch_create(g2pbase);

    return GSV_OK;
}

int gsv_load_base(gsv_engine *e, const char *manifest_path) {
    if (!e || !manifest_path || !manifest_path[0]) {
        if (e) gsv_set_error(e, "gsv_load_base: bad args");
        return GSV_ERR_INVALID_ARG;
    }
    /* single base: drop any previous before reloading */
    free_base(e);
    memset(&e->mf, 0, sizeof(e->mf));
    if (gsv_manifest_load(manifest_path, &e->mf) != 0) {
        gsv_set_error(e, "gsv_load_base: cannot parse %s", manifest_path);
        return GSV_ERR_IO;
    }
    if (e->mf.contract != 1) {
        gsv_set_error(e, "gsv_load_base: unsupported contract version %d", e->mf.contract);
        gsv_manifest_free(&e->mf);
        return GSV_ERR_GENERIC;
    }
    if (!version_supported(e->mf.tts_version)) {
        gsv_set_error(e, "gsv_load_base: unsupported gsv version '%s'", e->mf.tts_version);
        gsv_manifest_free(&e->mf);
        return GSV_ERR_GENERIC;
    }
    derive_dirname(manifest_path, e->pkg_dir, sizeof(e->pkg_dir));

    /* env (needed for session creation) */
    if (!e->env) {
        OrtStatus *st = e->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "gsvtts", &e->env);
        if (st) { gsv_set_error(e, "CreateEnv failed"); e->api->ReleaseStatus(st); return GSV_ERR_BACKEND; }
    }

    /* resolve device ONCE (cuda->cpu fallback); all character loads reuse e->active_device */
    char device[16];
    gsv_resolve_device(e, device, sizeof(device));
    e->cuda_fallback = 0;
    int rc = load_base_sessions(e, device);
    if (rc != GSV_OK && strcmp(device, "cpu") != 0) {
        free_base(e);   /* drop partial base before retrying on CPU */
        gsv_set_error(e, "gsv_load_base: %s session failed, retrying on cpu", device);
        e->cuda_fallback = 1;
        rc = load_base_sessions(e, "cpu");
    }
    if (rc != GSV_OK) {
        free_base(e);
        gsv_manifest_free(&e->mf);
        return rc;
    }
    snprintf(e->active_device, sizeof(e->active_device), "%s", e->cuda_fallback ? "cpu" : device);
    e->has_base = 1;
    return GSV_OK;
}

/* ---------------- character load (hot-pluggable) ---------------- */
static int gsv_load_name(gsv_engine *e, const char *sovits_name, const char *gpt_name) {
    if (!e) {
        gsv_set_error(e, "gsv_load_model: bad args");
        return GSV_ERR_INVALID_ARG;
    }
    if (!e->has_base) {
        gsv_set_error(e, "gsv_load_model: no base loaded (pass base_manifest at init)");
        return GSV_ERR_NOT_LOADED;
    }
    /* 空名字 = 该侧直接用底模回退 */
    if (!sovits_name) sovits_name = "";
    if (!gpt_name) gpt_name = "";
    /* single-resident: drop the previous character (base persists) */
    free_character(e);

    const char *root = e->pkg_dir;
    const char *dev  = e->active_device;
    char p[768];

    /* GPT (fp16 only): prefill + step, resolved by gpt_name with base fallback */
    if (!resolve_named(p, sizeof(p), root, e->mf.gpt_prefix, e->mf.gpt_fallback, gpt_name, "t2s_prefill_fp16.onnx")) {
        gsv_set_error(e, "gsv_load_model: GPT prefill not found for '%s'", gpt_name);
        return GSV_ERR_IO;
    }
    if (gsv_sess_create(e->api, e->env, p, dev, e->intra_op_threads, &e->t2s_prefill) != 0) {
        gsv_set_error(e, "gsv_load_model: ORT session create failed for GPT prefill (%s)", p);
        free_character(e); return GSV_ERR_BACKEND;
    }
    if (!resolve_named(p, sizeof(p), root, e->mf.gpt_prefix, e->mf.gpt_fallback, gpt_name, "t2s_step_fp16.onnx")) {
        gsv_set_error(e, "gsv_load_model: GPT step not found for '%s'", gpt_name);
        free_character(e); return GSV_ERR_IO;
    }
    if (gsv_sess_create(e->api, e->env, p, dev, e->intra_op_threads, &e->t2s_step) != 0) {
        gsv_set_error(e, "gsv_load_model: ORT session create failed for GPT step (%s)", p);
        free_character(e); return GSV_ERR_BACKEND;
    }

    /* SoVITS (device variant): enc_p + flowdec, resolved by sovits_name with base fallback */
    char stem[64];
    snprintf(stem, sizeof(stem), "enc_p_%s.onnx", sovits_variant(e, dev));
    if (!resolve_named(p, sizeof(p), root, e->mf.sovits_prefix, e->mf.sovits_fallback, sovits_name, stem)) {
        gsv_set_error(e, "gsv_load_model: SoVITS enc_p not found for '%s'", sovits_name);
        free_character(e); return GSV_ERR_BACKEND;
    }
    if (gsv_sess_create(e->api, e->env, p, dev, e->intra_op_threads, &e->voice.enc_p) != 0) {
        gsv_set_error(e, "gsv_load_model: ORT session create failed for enc_p (%s)", p);
        free_character(e); return GSV_ERR_BACKEND;
    }
    snprintf(stem, sizeof(stem), "flowdec_%s.onnx", sovits_variant(e, dev));
    if (!resolve_named(p, sizeof(p), root, e->mf.sovits_prefix, e->mf.sovits_fallback, sovits_name, stem)) {
        gsv_set_error(e, "gsv_load_model: SoVITS flowdec not found for '%s'", sovits_name);
        free_character(e); return GSV_ERR_IO;
    }
    if (gsv_sess_create(e->api, e->env, p, dev, e->intra_op_threads, &e->voice.flowdec) != 0) {
        gsv_set_error(e, "gsv_load_model: ORT session create failed for flowdec (%s)", p);
        free_character(e); return GSV_ERR_BACKEND;
    }

    /* version-level SoVITS params + identity */
    e->voice.cfg = e->mf.base_voice;
    if (sovits_name[0] && gpt_name[0]) snprintf(e->voice.name, sizeof(e->voice.name), "%s/%s", sovits_name, gpt_name);
    else snprintf(e->voice.name, sizeof(e->voice.name), "%s", sovits_name[0] ? sovits_name : gpt_name);
    e->has_voice = 1;
    return GSV_OK;
}

int gsv_load_model(gsv_engine *e, const char *sovits_name, const char *gpt_name) {
    if (!e) return GSV_ERR_INVALID_ARG;
    pthread_mutex_lock(&e->infer_lock);
    int rc = gsv_load_name(e, sovits_name, gpt_name);
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

int gsv_unload_model(gsv_engine *e) {
    if (!e) return GSV_ERR_INVALID_ARG;
    pthread_mutex_lock(&e->infer_lock);
    int rc = GSV_OK;
    if (!e->has_voice) {
        gsv_set_error(e, "gsv_unload_model: no model loaded");
        rc = GSV_ERR_NOT_LOADED;
    } else {
        free_character(e);   /* base persists */
        e->has_voice = 0;
    }
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

/* Release all loaded models (character + base) — used by gsv_engine_destroy. */
void gsv_engine_release_models(gsv_engine *e) {
    if (!e) return;
    free_character(e);
    free_base(e);
    gsv_manifest_free(&e->mf);
}


