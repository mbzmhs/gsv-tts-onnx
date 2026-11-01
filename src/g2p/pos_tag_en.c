/* pos_tag_en.c - C port of NLTK PerceptronTagger (eng model). */
#include "pos_tag_en.h"
#include "en_util.h"
#include "utf8.h"

#define PT_START1 "-START-"
#define PT_START2 "-START2-"
#define PT_END1   "-END-"
#define PT_END2   "-END2-"

static int pt_isdigit_cp(uint32_t cp) { return cp >= '0' && cp <= '9'; }

/* PerceptronTagger.normalize (input is accent-stripped; ASCII semantics). */
static const char *pt_normalize(const char *word, char *out, size_t outsz) {
    if (strchr(word, '-') && word[0] != '-') { snprintf(out, outsz, "!HYPHEN"); return out; }

    int all_digit = 1, ncp = 0;
    const char *p = word; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) { if (!pt_isdigit_cp(cp)) all_digit = 0; ncp++; }
    if (ncp == 4 && all_digit) { snprintf(out, outsz, "!YEAR"); return out; }
    const char *q = word; uint32_t c0 = gsv_utf8_decode(&q);
    if (c0 && pt_isdigit_cp(c0)) { snprintf(out, outsz, "!DIGITS"); return out; }

    size_t o = 0;
    p = word;
    while ((cp = gsv_utf8_decode(&p)) != 0 && o + 4 < outsz) {
        if (cp >= 'A' && cp <= 'Z') cp = cp - 'A' + 'a';
        int e = gsv_utf8_encode(cp, out + o); o += (size_t)e;
    }
    out[o] = 0;
    return out;
}

/* last up-to-3 codepoints of s into out (NUL-term). */
static void pt_last3(const char *s, char *out, size_t outsz) {
    const char *ring[3] = {s, s, s}; int n = 0;
    const char *p = s; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        int sz = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
        ring[n % 3] = p - sz;          /* start of this codepoint */
        n++;
    }
    int start_idx = (n > 3) ? ((n - 3) % 3) : 0;   /* oldest of last min(3,n) */
    const char *r = ring[start_idx];
    size_t o = 0;
    while (*r && o + 1 < outsz) out[o++] = *r++;
    out[o] = 0;
}

static void pt_predict(const gsv_pos_tagger *t, const char **feats, int n_feat, const char **best_out) {
    double *scores = (double *)calloc((size_t)(t->n_classes > 0 ? t->n_classes : 1), sizeof(double));
    if (!scores) { *best_out = t->classes[0]; return; }

    for (int fi = 0; fi < n_feat; fi++) {
        uint32_t idx = gsv_strmap_get(&t->weights, feats[fi], (int)strlen(feats[fi]));
        if (idx == UINT32_MAX || idx >= (uint32_t)t->n_feat) continue;
        int ne = t->f_nent[idx];
        for (int k = 0; k < ne; k++) {
            uint32_t e = (uint32_t)t->f_eoff[idx] + (uint32_t)k;
            int tag = t->f_enttag[e];
            if (tag < t->n_classes) scores[tag] += 1.0 * t->f_entwgt[e];
        }
    }
    int best = 0;
    for (int i = 1; i < t->n_classes; i++) {
        if (scores[i] > scores[best] ||
            (scores[i] == scores[best] && strcmp(t->classes[i], t->classes[best]) > 0))
            best = i;
    }
    *best_out = t->classes[best];
    free(scores);
}

void gsv_pos_tag(const gsv_pos_tagger *t, const char **words, int n_words, const char **tags_out) {
    const char **ctx = (const char **)malloc(sizeof(char *) * (size_t)(n_words + 4));
    char *normbuf = (char *)malloc((size_t)(n_words + 4) * 96);
    if (!ctx || !normbuf) { free(ctx); free(normbuf); return; }
    int nc = 0;
    ctx[nc++] = (char *)PT_START1;
    ctx[nc++] = (char *)PT_START2;
    for (int i = 0; i < n_words; i++)
        ctx[nc++] = pt_normalize(words[i], normbuf + (size_t)i * 96, 96);
    ctx[nc++] = (char *)PT_END1;
    ctx[nc++] = (char *)PT_END2;

    static char feat_store[15][96];
    const char *prev = PT_START1, *prev2 = PT_START2;
    for (int i = 0; i < n_words; i++) {
        const char *word = words[i];
        uint32_t ti = gsv_strmap_get(&t->tagdict, word, (int)strlen(word));
        const char *tag;
        if (ti != UINT32_MAX && ti < (uint32_t)t->n_classes) {
            tag = t->classes[ti];
        } else {
            int ci = i + 2;
            int nf = 0;
            char l3[96], l3b[96], l3c[96], p1[8] = {0};

            snprintf(feat_store[nf], sizeof feat_store[nf], "bias"); nf++;
            pt_last3(word, l3, sizeof l3);
            snprintf(feat_store[nf], sizeof feat_store[nf], "i suffix %s", l3); nf++;
            const char *qw = word; uint32_t c0 = gsv_utf8_decode(&qw);
            if (c0) gsv_utf8_encode(c0, p1);
            snprintf(feat_store[nf], sizeof feat_store[nf], "i pref1 %s", p1); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i-1 tag %s", prev); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i-2 tag %s", prev2); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i tag+i-2 tag %s %s", prev, prev2); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i word %s", ctx[ci]); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i-1 tag+i word %s %s", prev, ctx[ci]); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i-1 word %s", ctx[ci - 1]); nf++;
            pt_last3(ctx[ci - 1], l3b, sizeof l3b);
            snprintf(feat_store[nf], sizeof feat_store[nf], "i-1 suffix %s", l3b); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i-2 word %s", ctx[ci - 2]); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i+1 word %s", ctx[ci + 1]); nf++;
            pt_last3(ctx[ci + 1], l3c, sizeof l3c);
            snprintf(feat_store[nf], sizeof feat_store[nf], "i+1 suffix %s", l3c); nf++;
            snprintf(feat_store[nf], sizeof feat_store[nf], "i+2 word %s", ctx[ci + 2]); nf++;

            const char *fptrs[15];
            for (int k = 0; k < nf; k++) fptrs[k] = feat_store[k];
            pt_predict(t, fptrs, nf, &tag);
        }
        tags_out[i] = tag;
        prev2 = prev;
        prev = tag;
    }
    free(ctx); free(normbuf);
}

/* ---- load ---- */
int gsv_pos_tag_load(gsv_pos_tagger *t, const char *dir) {
    memset(t, 0, sizeof *t);
    size_t len = 0;
    char *buf = gsv_en_read_file(dir, "pos_tag.bin", &len);
    if (!buf || len < 8) { free(buf); return -1; }
    if (rd_u32(buf) != 0x47545050) { free(buf); return -2; }
    const char *p = buf + 4;

    int nc = (int)rd_u32(p); p += 4;
    t->n_classes = nc;
    t->classes = (char **)malloc(sizeof(char *) * (size_t)(nc > 0 ? nc : 1));
    if (!t->classes) return -3;
    for (int i = 0; i < nc; i++) {
        int l = (unsigned char)p[0]; p++;
        t->classes[i] = (char *)malloc((size_t)l + 1);
        memcpy(t->classes[i], p, (size_t)l); t->classes[i][l] = 0;
        p += l;
    }

    int nt = (int)rd_u32(p); p += 4;
    if (gsv_strmap_init(&t->tagdict, nt) != 0) return -4;
    for (int i = 0; i < nt; i++) {
        int wl = rd_u16(p); p += 2;
        uint16_t tag = rd_u16(p + wl); p += wl + 2;
        if (gsv_strmap_put(&t->tagdict, p - wl - 2, wl, tag) != 0) return -5;
    }

    int nf = (int)rd_u32(p); p += 4;
    t->n_feat = nf;
    if (gsv_strmap_init(&t->weights, nf) != 0) return -6;
    t->f_nent = (uint16_t *)calloc((size_t)(nf > 0 ? nf : 1), 2);
    t->f_eoff = (uint32_t *)calloc((size_t)(nf > 0 ? nf : 1), 4);
    if (!t->f_nent || !t->f_eoff) return -7;
    {
        const char *q = p; uint32_t tot = 0;
        for (int i = 0; i < nf; i++) {
            int fl = rd_u16(q); q += 2 + fl;
            int ne = rd_u16(q); q += 2 + (size_t)ne * 10;
            tot += (uint32_t)ne;
        }
        t->f_enttag = (uint8_t *)malloc((size_t)(tot > 0 ? tot : 1));
        t->f_entwgt = (double *)calloc((size_t)(tot > 0 ? tot : 1), sizeof(double));
        if (!t->f_enttag || !t->f_entwgt) return -8;
    }
    uint32_t eoff = 0;
    for (int i = 0; i < nf; i++) {
        int fl = rd_u16(p); p += 2;
        const char *featp = p;
        int ne = rd_u16(p + fl); p += fl + 2;
        t->f_nent[i] = (uint16_t)ne;
        t->f_eoff[i] = eoff;
        for (int k = 0; k < ne; k++) {
            uint16_t tag = rd_u16(p); p += 2;
            double wgt = rd_f64(p); p += 8;
            t->f_enttag[eoff] = (uint8_t)tag;
            t->f_entwgt[eoff] = wgt;
            eoff++;
        }
        if (gsv_strmap_put(&t->weights, featp, fl, (uint32_t)i) != 0) return -9;
    }
    free(buf);
    return 0;
}

void gsv_pos_tag_free(gsv_pos_tagger *t) {
    for (int i = 0; i < t->n_classes; i++) free(t->classes[i]);
    free(t->classes);
    gsv_strmap_free(&t->tagdict);
    gsv_strmap_free(&t->weights);
    free(t->f_nent); free(t->f_eoff); free(t->f_enttag); free(t->f_entwgt);
    memset(t, 0, sizeof *t);
}
