/* bert_tok.c - RoBERTa WordPiece tokenizer + per-phone BERT features (M6b, zh). */
#include "bert_tok.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "g2p/utf8.h"

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}

int gsv_bert_tok_load(gsv_bert_tok *t, const char *path) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = (char *)malloc((size_t)sz + 1);
    if (!data || fread(data, 1, (size_t)sz, f) != (size_t)sz) { free(data); fclose(f); return -1; }
    data[sz] = 0; fclose(f);

    int n = 0;
    for (char *p = data; *p;) { char *nl = strchr(p, '\n'); if (nl) p = nl + 1; else break; n++; }
    t->n = n;
    t->tok = (char **)malloc(sizeof(char *) * (size_t)n);
    gsv_buf ar; memset(&ar, 0, sizeof ar);
    /* Pre-size the arena to the whole file so it never reallocs during interning.
     * t->tok[i] stores pointers INTO this buffer; a mid-loop realloc would dangle
     * them (freed memory looked valid at load time but is clobbered later). */
    if (gsv_buf_reserve(&ar, (size_t)sz + 1) != 0) { free(t->tok); free(data); return -1; }
    char *line = data;
    for (int i = 0; i < n; i++) {
        char *nl = strchr(line, '\n');
        size_t ln = nl ? (size_t)(nl - line) : strlen(line);
        if (ln > 0 && line[ln - 1] == '\r') ln--;
        t->tok[i] = gsv_buf_intern(&ar, line, ln);
        if (!t->tok[i]) { free(ar.data); free(t->tok); free(data); return -1; }
        if (nl) line = nl + 1; else break;
    }

    int cap = 1; while (cap < n * 2) cap <<= 1;
    t->cap = cap;
    t->hash = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)cap);
    t->idx  = (int *)malloc(sizeof(int) * (size_t)cap);
    if (!t->hash || !t->idx) { free(ar.data); free(t->tok); free(t->hash); free(t->idx); free(data); return -1; }
    for (int i = 0; i < cap; i++) { t->hash[i] = 0; t->idx[i] = -1; }
    for (int i = 0; i < n; i++) {
        uint32_t h = fnv1a(t->tok[i]);
        uint32_t slot = h & (uint32_t)(cap - 1);
        while (t->idx[slot] != -1) slot = (slot + 1) & (uint32_t)(cap - 1);
        t->hash[slot] = h; t->idx[slot] = i;
    }
    t->arena = ar.data;
    t->cls = gsv_bert_tok_id(t, "[CLS]");
    t->sep = gsv_bert_tok_id(t, "[SEP]");
    t->unk = gsv_bert_tok_id(t, "[UNK]");
    free(data);
    return (t->cls >= 0 && t->sep >= 0 && t->unk >= 0) ? 0 : -1;
}

void gsv_bert_tok_free(gsv_bert_tok *t) {
    if (!t) return;
    free(t->tok); free(t->hash); free(t->idx); free(t->arena);
    memset(t, 0, sizeof(*t));
}

int gsv_bert_tok_id(const gsv_bert_tok *t, const char *tok) {
    if (!t || !t->n || !tok) return -1;
    uint32_t h = fnv1a(tok);
    uint32_t slot = h & (uint32_t)(t->cap - 1);
    for (;;) {
        int i = t->idx[slot];
        if (i == -1) return -1;
        if (t->hash[slot] == h && strcmp(t->tok[i], tok) == 0) return i;
        slot = (slot + 1) & (uint32_t)(t->cap - 1);
    }
}

/* HF _is_chinese_char ranges. */
static int gsv_is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

static int gsv_is_alnum(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9');
}

/* WordPiece greedy longest-prefix on a single basic-word (bytes, length wl). */
static void wp_word(const gsv_bert_tok *t, const char *w, int wl, int64_t *out, int cap, int *n) {
    int i = 0;
    int is_first = 1;
    while (i < wl) {
        int best_len = 0, best_id = -1;
        for (int j = wl; j > i; j--) {
            int slen = j - i;
            char cand[80];
            int off = 0;
            if (!is_first) { cand[off++] = '#'; cand[off++] = '#'; }
            memcpy(cand + off, w + i, (size_t)slen); off += slen; cand[off] = 0;
            int id = gsv_bert_tok_id(t, cand);
            if (id >= 0) { best_len = slen; best_id = id; break; }
        }
        if (best_len == 0) {
            if (*n < cap) out[(*n)++] = (int64_t)t->unk;
            const char *pp = w + i; uint32_t c; (void)c; gsv_utf8_decode(&pp); i = (int)(pp - w);
        } else {
            if (*n < cap) out[(*n)++] = (int64_t)best_id;
            i += best_len;
        }
        is_first = 0;
    }
}

int gsv_bert_tokenize(const gsv_bert_tok *t, const char *text, int64_t *out,
                      int cap, int *n_tokens) {
    if (!t || !t->n || !text || !out || !n_tokens) return -1;
    char word[64]; int wl = 0;
    int n = 0;
    if (n < cap) out[n++] = (int64_t)t->cls;

    const char *p = text; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        if (gsv_is_cjk(cp)) {
            if (wl) { wp_word(t, word, wl, out, cap, &n); wl = 0; }
            char cb[4]; int cl = gsv_utf8_encode(cp, cb);
            wp_word(t, cb, cl, out, cap, &n);
        } else if (gsv_is_alnum(cp)) {
            if (wl < (int)sizeof(word) - 1) word[wl++] = (char)cp;
            else { wp_word(t, word, wl, out, cap, &n); word[wl = 0] = (char)cp; wl = 1; }
        } else {
            if (wl) { wp_word(t, word, wl, out, cap, &n); wl = 0; }
            if (cp != ' ' && cp != 0x3000) {   /* skip ASCII + ideographic space */
                char cb[4]; int cl = gsv_utf8_encode(cp, cb);
                wp_word(t, cb, cl, out, cap, &n);
            }
        }
    }
    if (wl) wp_word(t, word, wl, out, cap, &n);
    if (n < cap) out[n++] = (int64_t)t->sep;
    *n_tokens = n;
    return 0;
}

int gsv_bert_phone_features(gsv_engine *e, char **words, int n_words,
                            const int *phs, int n_phones, float *out) {
    if (!e || !e->has_base) return GSV_ERR_NOT_LOADED;
    if (!e->bert_tok.n || !words || !phs || !out) return GSV_ERR_INVALID_ARG;
    if (n_words <= 0 || n_phones <= 0) return GSV_ERR_INVALID_ARG;
    if (gsv_bert_ensure(e) != 0) return GSV_ERR_BACKEND;   /* lazily load zh bert.onnx */

    /* text = concatenation of word2ph words */
    gsv_buf tb; memset(&tb, 0, sizeof tb);
    for (int i = 0; i < n_words; i++) if (words[i]) gsv_buf_append_str(&tb, words[i]);
    if (gsv_buf_intern(&tb, "", 0) == NULL) { free(tb.data); return GSV_ERR_NO_MEMORY; }   /* NUL-terminate tb */

    int cap = n_words + 4;   /* content tokens <= n_words for single-char zh; +CLS/SEP */
    int64_t *ids  = (int64_t *)malloc(sizeof(int64_t) * (size_t)cap);
    int64_t *mask = (int64_t *)malloc(sizeof(int64_t) * (size_t)cap);
    if (!ids || !mask) { free(ids); free(mask); free(tb.data); return GSV_ERR_NO_MEMORY; }

    int T = 0;
    if (gsv_bert_tokenize(&e->bert_tok, tb.data, ids, cap, &T) != 0) {
        free(ids); free(mask); free(tb.data); return GSV_ERR_GENERIC;
    }
    const int maxlen = 512;   /* RoBERTa max_position_embeddings */
    if (T > maxlen) { free(ids); free(mask); free(tb.data); return GSV_ERR_GENERIC; }
    for (int i = 0; i < T; i++) mask[i] = 1;

    int64_t d2[2] = { 1, T };
    OrtValue *ix = gsv_tensor_create(e->api, ids, d2, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    OrtValue *im = gsv_tensor_create(e->api, mask, d2, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!ix || !im) {
        gsv_tensor_destroy(e->api, ix); gsv_tensor_destroy(e->api, im);
        free(ids); free(mask); free(tb.data); return GSV_ERR_BACKEND;
    }
    const void *in[2] = { ix, im };   /* input_ids, attention_mask */
    gsv_out o[1];
    int rc = gsv_sess_run(e->api, &e->bert, in, o, 1);
    gsv_tensor_destroy(e->api, ix); gsv_tensor_destroy(e->api, im);
    if (rc != 0) { free(ids); free(mask); free(tb.data); return GSV_ERR_BACKEND; }

    float *hs = (float *)o[0].data;   /* [1, T, 1024] */
    int dim = 1024;
    int n_content = T - 2;            /* drop [CLS] + [SEP] */
    if (n_content != n_words) {       /* alignment guard: one token per content char */
        gsv_sess_free_outputs(e->api, o, 1);
        free(ids); free(mask); free(tb.data); return GSV_ERR_GENERIC;
    }

    int p = 0;
    for (int w = 0; w < n_words; w++) {
        const float *cf = hs + (size_t)(w + 1) * dim;   /* skip CLS at [0] */
        for (int k = 0; k < phs[w]; k++) {
            memcpy(out + (size_t)p * dim, cf, (size_t)dim * sizeof(float));
            p++;
        }
    }

    gsv_sess_free_outputs(e->api, o, 1);
    free(ids); free(mask); free(tb.data);
    return GSV_OK;
}

GSVTTS_API int gsv_bert_token_ids(gsv_engine *e, const char *text, int64_t *out,
                       int cap, int *n_out) {
    if (!e || !e->bert_tok.n || !text || !out || !n_out) return GSV_ERR_INVALID_ARG;
    int T = 0;
    int rc = gsv_bert_tokenize(&e->bert_tok, text, out, cap, &T);
    if (rc != 0) return rc;
    *n_out = T;
    return GSV_OK;
}
