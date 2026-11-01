/* g2p_dispatch.c - language-dispatched G2P -> phone ids (M6). See g2p_dispatch.h. */
#include "g2p_dispatch.h"

#include <stdlib.h>
#include <string.h>

#include "g2p/g2p_zh.h"
#include "g2p/g2p_en.h"
#include "g2p/ja/g2p_ja.h"
#include "g2p/symbols.h"
#include "g2p/utf8.h"

struct gsv_g2p_dispatch {
    int has_zh, has_en, has_ja;
    gsv_g2p_full zh;     /* value: owns pinyin + jieba data */
    gsv_g2p_en_full en;  /* value: owns cmu/namedict/pos/gru/wseg data */
    gsv_g2p_ja *ja;      /* heap: owns OpenJTalk objects */
};

/* ---------------- pause map (Pause.py) - shared post-processing for ja -------- */
static int is_pause_cp(uint32_t cp) {
    switch (cp) {
        case 0x21: case 0xFF01:   /* ! ！ */
        case 0x3F: case 0xFF1F:   /* ? ？ */
        case 0x2026: case 0x7E:   /* … ~ */
        case 0x2E: case 0x3002:   /* . 。 */
        case 0x2C: case 0xFF0C:   /* , ， */
        case 0x3A: case 0xFF1A:   /* : ： */
        case 0x3B: case 0xFF1B:   /* ; ； */
        case 0x3001: case 0x30FB: /* 、 ・ */
            return 1;
    }
    return 0;
}
static const char *pause_symbol_cp(uint32_t cp) {
    switch (cp) {
        case 0x21: case 0xFF01: return "!";
        case 0x3F: case 0xFF1F: return "?";
        case 0x2026: case 0x7E: return "\xE2\x80\xA6"; /* … */
        case 0x2E: case 0x3002: return ".";
        case 0x2C: case 0xFF0C: case 0x3A: case 0xFF1A:
        case 0x3B: case 0xFF1B: case 0x3001: case 0x30FB: return ",";
    }
    return NULL;
}
static int is_punct_sym(const char *s) {
    for (int k = 0; k < GSV_N_PUNCTUATION; k++)
        if (!strcmp(gsv_punctuation[k], s)) return 1;
    return 0;
}

/* Map an already-final phone symbol list (zh/en *_to_phonemes output) to ids.
 * Returns count, or -1 if out has fewer than cap... i.e. if it overflows cap. */
static int map_final_to_ids(const char **phones, int n, int64_t *out, int cap) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        int id = gsv_symbol_to_id(phones[i]);
        if (id < 0) continue;   /* defensive: post-processing should have removed these */
        if (count >= cap) return -1;
        out[count++] = (int64_t)id;
    }
    return count;
}

/* Apply the shared __init__.py post-processing (pause remap + UNK + dedup) to a
 * raw ja phone list and emit ids. Returns count, or -1 on overflow. */
static int ja_raw_to_ids(const char **phones, int n, int64_t *out, int cap) {
    int count = 0;
    int prev_id = -1;   /* id of the previous valid effective symbol, else -1 */
    for (int i = 0; i < n; i++) {
        const char *sym = phones[i];
        const char *p = sym;
        uint32_t cp = gsv_utf8_decode(&p);
        if (is_pause_cp(cp)) sym = pause_symbol_cp(cp);   /* pause remap */
        int id = gsv_symbol_to_id(sym);                   /* UNK if not in set */
        if (id < 0) { prev_id = -1; continue; }           /* drop UNK, reset chain */
        int dup = (prev_id == id) && is_punct_sym(sym);   /* consecutive punct dedup */
        if (!dup) {
            if (count >= cap) return -1;
            out[count++] = (int64_t)id;
        }
        prev_id = id;
    }
    return count;
}

/* __init__.py: re.sub(r'\.{3,}|。{3,}', '…', text). Runs of >=3 '.' or '。' -> "…". */
static int dot_collapse(const char *in, char *out, int outcap) {
    int o = 0;
    const char *p = in;
    while (*p) {
        uint32_t cp = gsv_utf8_decode(&p);
        if (cp == 0x2E || cp == 0x3002) {   /* '.' or '。' */
            /* run counts the current dot PLUS following ones. The outer decode above
             * already advanced p past the current dot, so r=p points at the char AFTER
             * it; start run=1 to include the current one (else a lone dot is dropped). */
            int run = 1;
            const char *r = p;
            uint32_t c;
            /* `*r &&` guard: gsv_utf8_decode advances past a NUL (returns 0 but
             * increments), so without it a trailing dot would let `r`/`p` run past
             * the string end into following memory (OOB read). */
            while (*r && (c = gsv_utf8_decode(&r)) == cp) { if (++run > 1000000) break; }
            if (run >= 3) {
                if (o + 3 > outcap) return -1;
                out[o++] = 0xE2; out[o++] = 0x80; out[o++] = 0xA6;   /* … */
            } else {
                char b[4]; int bn = gsv_utf8_encode(cp, b);
                for (int k = 0; k < run; k++) {
                    if (o + bn > outcap) return -1;
                    memcpy(out + o, b, (size_t)bn); o += bn;
                }
            }
            p = r;
        } else {
            char b[4]; int bn = gsv_utf8_encode(cp, b);
            if (o + bn > outcap) return -1;
            memcpy(out + o, b, (size_t)bn); o += bn;
        }
    }
    if (o >= outcap) return -1;
    out[o] = '\0';
    return 0;
}

/* ---------------- lifecycle ---------------- */
gsv_g2p_dispatch *gsv_g2p_dispatch_create(const char *base_dir) {
    if (!base_dir || !base_dir[0]) return NULL;
    gsv_g2p_dispatch *d = (gsv_g2p_dispatch *)calloc(1, sizeof(*d));
    if (!d) return NULL;

    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/zh", base_dir);
    if (gsv_g2p_full_load(&d->zh, dir) == 0) d->has_zh = 1;

    snprintf(dir, sizeof(dir), "%s/en", base_dir);
    if (gsv_g2p_en_load(&d->en, dir) == 0) d->has_en = 1;

    snprintf(dir, sizeof(dir), "%s/ja", base_dir);
    d->ja = gsv_g2p_ja_create(dir);
    if (d->ja) d->has_ja = 1;

    return d;
}

void gsv_g2p_dispatch_destroy(gsv_g2p_dispatch *d) {
    if (!d) return;
    if (d->has_zh) gsv_g2p_full_free(&d->zh);
    if (d->has_en) gsv_g2p_en_free(&d->en);
    if (d->ja) gsv_g2p_ja_destroy(d->ja);
    free(d);
}

int gsv_g2p_dispatch_has(const gsv_g2p_dispatch *d, const char *lang) {
    if (!d || !lang) return 0;
    if (!strcmp(lang, "zh")) return d->has_zh;
    if (!strcmp(lang, "en")) return d->has_en;
    if (!strcmp(lang, "ja")) return d->has_ja;
    return 0;
}

/* ---------------- dispatch ---------------- */
int gsv_g2p_dispatch_ids(const gsv_g2p_dispatch *d, const char *text,
                         const char *lang, int64_t *out_ids, int cap, int *out_n) {
    if (!d || !text || !lang || !out_ids || !out_n || cap < 0) return GSV_ERR_INVALID_ARG;
    *out_n = 0;

    if (!strcmp(lang, "zh")) {
        if (!d->has_zh) return GSV_ERR_NOT_LOADED;
        gsv_g2p_result r;
        if (gsv_g2p_full_to_phonemes(&d->zh, text, &r) != 0) return GSV_ERR_GENERIC;
        int c = map_final_to_ids((const char **)r.phones, r.n_phones, out_ids, cap);
        if (c < 0) return GSV_ERR_NO_MEMORY;
        *out_n = c;
        return GSV_OK;
    }

    if (!strcmp(lang, "en")) {
        if (!d->has_en) return GSV_ERR_NOT_LOADED;
        gsv_g2p_en_result r;
        if (gsv_g2p_en_to_phonemes(&d->en, text, &r) != 0) return GSV_ERR_GENERIC;
        int c = map_final_to_ids((const char **)r.phones, r.n_phones, out_ids, cap);
        if (c < 0) return GSV_ERR_NO_MEMORY;
        *out_n = c;
        return GSV_OK;
    }

    if (!strcmp(lang, "ja")) {
        if (!d->has_ja || !d->ja) return GSV_ERR_NOT_LOADED;
        char collapsed[8192];
        if (dot_collapse(text, collapsed, (int)sizeof(collapsed)) != 0) return GSV_ERR_NO_MEMORY;
        gsv_g2p_ja_result r;
        if (gsv_g2p_ja_g2p(d->ja, collapsed, &r) != 0) return GSV_ERR_GENERIC;
        int c = ja_raw_to_ids((const char **)r.phones, r.n_phones, out_ids, cap);
        if (c < 0) return GSV_ERR_NO_MEMORY;
        *out_n = c;
        return GSV_OK;
    }

    return GSV_ERR_INVALID_ARG;   /* unsupported / "auto" */
}

/* ---------------- full result (ids + word2ph) for BERT alignment ---------------- */
void gsv_g2p_full_result_free(gsv_g2p_full_result *r) {
    if (!r) return;
    free(r->ids); r->ids = NULL; r->n_ids = 0;
    if (r->words) { for (int i = 0; i < r->n_words; i++) free(r->words[i]); free(r->words); }
    r->words = NULL; r->n_words = 0;
    free(r->phs); r->phs = NULL; r->n_phs = 0;
}

int gsv_g2p_dispatch_full(const gsv_g2p_dispatch *d, const char *text, const char *lang,
                          gsv_g2p_full_result *out) {
    if (!d || !text || !lang || !out) return GSV_ERR_INVALID_ARG;
    out->ids = NULL; out->n_ids = 0;
    out->words = NULL; out->n_words = 0; out->phs = NULL; out->n_phs = 0;

    if (!strcmp(lang, "zh")) {
        if (!d->has_zh) return GSV_ERR_NOT_LOADED;
        gsv_g2p_result r;
        if (gsv_g2p_full_to_phonemes(&d->zh, text, &r) != 0) return GSV_ERR_GENERIC;
        int n = r.n_phones;
        out->ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
        if (!out->ids) return GSV_ERR_NO_MEMORY;
        /* Defensive: drop invalid symbols (id<0, e.g. emoji-only text where G2P emits
         * placeholders) instead of keeping them as id 0 — an all-zeros target phone
         * sequence reaches t2s and corrupts the heap. Dropping may leave n_ids == 0,
         * which the caller reports as GSV_ERR_TEXT_EMPTY. Matches map_final_to_ids(). */
        int n_valid = 0;
        for (int i = 0; i < n; i++) {
            int id = gsv_symbol_to_id(r.phones[i]);
            if (id < 0) continue;
            out->ids[n_valid++] = (int64_t)id;
        }
        out->n_ids = n_valid;
        if (r.n_words > 0) {
            out->words = (char **)malloc(sizeof(char *) * (size_t)r.n_words);
            if (!out->words) { gsv_g2p_full_result_free(out); return GSV_ERR_NO_MEMORY; }
            for (int i = 0; i < r.n_words; i++) {
                size_t L = strlen(r.words[i]) + 1;
                char *c = (char *)malloc(L);
                if (!c) { gsv_g2p_full_result_free(out); return GSV_ERR_NO_MEMORY; }
                memcpy(c, r.words[i], L); out->words[i] = c;
            }
            out->n_words = r.n_words;
        }
        if (r.n_phs > 0) {
            out->phs = (int *)malloc(sizeof(int) * (size_t)r.n_phs);
            if (!out->phs) { gsv_g2p_full_result_free(out); return GSV_ERR_NO_MEMORY; }
            memcpy(out->phs, r.phs, sizeof(int) * (size_t)r.n_phs);
            out->n_phs = r.n_phs;
        }
        return GSV_OK;
    }

    if (!strcmp(lang, "en") || !strcmp(lang, "ja")) {
        /* BERT is zeros for these langs: ids only. */
        int cap = (int)strlen(text) * 12 + 512;
        for (int attempt = 0; attempt < 4; attempt++) {
            int64_t *ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)cap);
            if (!ids) return GSV_ERR_NO_MEMORY;
            int n = 0;
            int rc = gsv_g2p_dispatch_ids(d, text, lang, ids, cap, &n);
            if (rc == GSV_OK) { out->ids = ids; out->n_ids = n; return GSV_OK; }
            free(ids);
            if (rc != GSV_ERR_NO_MEMORY) return rc;
            cap *= 2;
        }
        return GSV_ERR_NO_MEMORY;
    }

    return GSV_ERR_INVALID_ARG;
}
