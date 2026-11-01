/* g2p_zh.c - C port of the zh G2P orchestration (chinese.py + __init__.py).
 * Memory model: result STRINGS live in g->arena (reset per call); the pointer arrays
 * (phones/words/phs) are malloc'd and tracked in keep_* (freed on the next call). */
#include "g2p_zh.h"
#include "tone_sandhi.h"
#include "text_norm.h"
#include "symbols.h"
#include "erhua_sets.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- pause map (Pause.py) ---------------- */
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

/* ---------------- dynamic accumulators (arrays malloc'd; strings live in arena) ---- */
typedef struct { char **a; int n, cap; } strarr;
static void sa_push(strarr *s, const char *v) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 16; s->a = (char **)realloc(s->a, (size_t)s->cap * sizeof(char *)); }
    s->a[s->n++] = (char *)v;
}
typedef struct { int *a; int n, cap; } intarr;
static void ia_push(intarr *s, int v) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 16; s->a = (int *)realloc(s->a, (size_t)s->cap * sizeof(int)); }
    s->a[s->n++] = v;
}

static int in_strset(const char *s, const char **set, int n) {
    for (int i = 0; i < n; i++) if (strcmp(set[i], s) == 0) return 1;
    return 0;
}

/* opencpop lookup */
static int oc_lookup(gsv_g2p_full *g, const char *pin, int plen, const char **c, const char **v) {
    for (int i = 0; i < g->n_oc; i++) {
        if (g->oc_klen[i] == (uint32_t)plen && memcmp(g->oc_key[i], pin, (size_t)plen) == 0) { *c = g->oc_c[i]; *v = g->oc_v[i]; return 1; }
    }
    return 0;
}

/* ---------------- _merge_erhua (chinese.py) ---------------- */
static void merge_erhua(char **fin, int n, const char *word, const char *pos) {
    int wlen = gsv_utf8_len(word);
    uint32_t *wc = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(wlen + 1));
    int wn = 0; const char *p = word; uint32_t c;
    while ((c = gsv_utf8_decode(&p)) != 0) wc[wn++] = c;

    if (n > 0 && wn == n && wc[wn - 1] == 0x513F && strcmp(fin[n - 1], "er1") == 0)
        strcpy(fin[n - 1], "er2");

    int in_must = in_strset(word, gsv_must_erhua, GSV_N_MUST_ERHUA);
    int in_not  = in_strset(word, gsv_not_erhua, GSV_N_NOT_ERHUA);
    int pos_bad = (strcmp(pos, "a") == 0 || strcmp(pos, "j") == 0 || strcmp(pos, "nr") == 0);
    if (!in_must && (in_not || pos_bad)) { free(wc); return; }
    if (n != wlen) { free(wc); return; }

    /* last 2 chars of word as UTF-8 (only meaningful when wn>=2) */
    int last2_in_not = 0;
    if (wn >= 2) {
        char l2[8]; int l2n = 0;
        for (int i = wn - 2; i < wn; i++) l2n += gsv_utf8_encode(wc[i], l2 + l2n);
        l2[l2n] = 0;
        last2_in_not = in_strset(l2, gsv_not_erhua, GSV_N_NOT_ERHUA);
    }

    for (int i = 0; i < n; i++) {
        if (i == n - 1 && wc[i] == 0x513F &&
            (strcmp(fin[i], "er2") == 0 || strcmp(fin[i], "er5") == 0) &&
            !last2_in_not && i > 0) {
            char nb[8];
            snprintf(nb, sizeof nb, "er%s", fin[i - 1] + strlen(fin[i - 1]) - 1);
            strcpy(fin[i], nb);
        }
    }
    free(wc);
}

/* intern a copy of a pypinyin outbuf string into the arena (outbuf resets per lazy call) */
static char *arena_dup(gsv_g2p_full *g, const char *s) {
    return gsv_buf_intern(&g->arena, s, strlen(s));
}
static char *arena_dup_bytes(gsv_g2p_full *g, const char *b, int n) {
    return gsv_buf_intern(&g->arena, b, (size_t)n);
}

/* ---------------- per-segment _g2p core ---------------- */
static void process_segment(gsv_g2p_full *g, const uint32_t *cps, int s, int e,
                            strarr *ph, strarr *wd, intarr *ps) {
    gsv_g2p_zh *py = &g->py;
    gsv_jieba  *jb = &g->jb;

    /* 1. remove [a-zA-Z]+ -> filtered cps */
    uint32_t *fc = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(e - s + 1));
    int fn = 0;
    for (int i = s; i < e; i++) {
        uint32_t c = cps[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) continue;
        fc[fn++] = c;
    }

    /* 2. lcut */
    gsv_seg *segs = (gsv_seg *)malloc(sizeof(gsv_seg) * (size_t)(2 * fn + 8));
    int sn = gsv_posseg_lcut(jb, fc, fn, segs, 2 * fn + 8);

    /* 3. materialize input words/tags into a pre-sized buf (no realloc during loop) */
    gsv_buf inbuf; memset(&inbuf, 0, sizeof inbuf);
    {
        size_t need = 16; char eb[4];
        for (int i = 0; i < sn; i++) {
            int t = 0; for (uint32_t k = segs[i].start; k < segs[i].end; k++) t += gsv_utf8_encode(fc[k], eb);
            need += (size_t)t + 1 + strlen(segs[i].tag) + 1;
        }
        gsv_buf_reserve(&inbuf, need);
    }
    gsv_ts_seg *in = (gsv_ts_seg *)malloc(sizeof(gsv_ts_seg) * (size_t)(sn + 4));
    for (int i = 0; i < sn; i++) {
        char tmp[512]; int t = 0;
        for (uint32_t k = segs[i].start; k < segs[i].end; k++) t += gsv_utf8_encode(fc[k], tmp + t);
        tmp[t] = 0;
        in[i].w = gsv_buf_intern(&inbuf, tmp, (size_t)t);
        in[i].p = gsv_buf_intern(&inbuf, segs[i].tag, strlen(segs[i].tag));
    }

    /* 4. pre_merge */
    gsv_buf ar; memset(&ar, 0, sizeof ar);
    gsv_ts_seg *merged = (gsv_ts_seg *)malloc(sizeof(gsv_ts_seg) * (size_t)(sn + 4));
    int mn = gsv_ts_pre_merge(jb, py, &ar, in, sn, merged, sn + 4);

    /* 5. per token: pinyin + tone + erhua -> symbols (word char aligned per syllable) */
    for (int i = 0; i < mn; i++) {
        const char *word = merged[i].w;
        const char *pos  = merged[i].p;
        if (strcmp(pos, "eng") == 0) continue;

        int wl = gsv_utf8_len(word);
        uint32_t *wcps = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(wl + 1));
        int wn = 0; const char *wp = word; uint32_t cc;
        while ((cc = gsv_utf8_decode(&wp)) != 0) wcps[wn++] = cc;

        char **itmp = (char **)malloc(sizeof(char *) * (size_t)(wl ? wl : 1));
        char **ftmp = (char **)malloc(sizeof(char *) * (size_t)(wl ? wl : 1));
        int ni = 0, nf = 0;
        gsv_pinyin_lazy(py, word, GSV_PY_STYLE_INITIALS, itmp, &ni);
        char **ini = (char **)malloc(sizeof(char *) * (size_t)(ni ? ni : 1));
        for (int k = 0; k < ni; k++) ini[k] = arena_dup(g, itmp[k]);
        gsv_pinyin_lazy(py, word, GSV_PY_STYLE_FINALS3, ftmp, &nf);
        char **fin = (char **)malloc(sizeof(char *) * (size_t)(nf ? nf : 1));
        for (int k = 0; k < nf; k++) fin[k] = arena_dup(g, ftmp[k]);

        gsv_ts_modified_tone(jb, word, pos, fin, nf);
        merge_erhua(fin, nf, word, pos);

        int ns = ni < nf ? ni : nf;
        for (int k = 0; k < ns; k++) {
            const char *ci = ini[k];
            const char *vf = fin[k];

            /* word char for this syllable (aligned with phs) */
            if (k < wn) { char cb[4]; int cn = gsv_utf8_encode(wcps[k], cb); sa_push(wd, arena_dup_bytes(g, cb, cn)); }

            if (strcmp(ci, vf) == 0) {              /* pause char: c == v -> phone=[c] */
                sa_push(ph, arena_dup(g, ci));
                ia_push(ps, 1);
                continue;
            }
            int vlen = (int)strlen(vf);
            if (vlen < 2) { ia_push(ps, 0); continue; }
            char tone = vf[vlen - 1];
            int vwt_len = vlen - 1;

            char pin[16]; int pn = 0;
            memcpy(pin, ci, strlen(ci)); pn = (int)strlen(ci);
            memcpy(pin + pn, vf, (size_t)vwt_len); pn += vwt_len; pin[pn] = 0;

            if (ci[0]) {                             /* multi-syllable: has initial */
                const char *rep = NULL;
                if (vwt_len == 3 && memcmp(vf, "uei", 3) == 0) rep = "ui";
                else if (vwt_len == 3 && memcmp(vf, "iou", 3) == 0) rep = "iu";
                else if (vwt_len == 3 && memcmp(vf, "uen", 3) == 0) rep = "un";
                if (rep) { int rl = (int)strlen(rep); memcpy(pin + pn - vwt_len, rep, (size_t)rl); pn -= vwt_len - rl; pin[pn] = 0; }
            } else {                                 /* single syllable: no initial */
                const char *full = NULL;
                if (strcmp(pin, "ing") == 0) full = "ying";
                else if (strcmp(pin, "i") == 0) full = "yi";
                else if (strcmp(pin, "in") == 0) full = "yin";
                else if (strcmp(pin, "u") == 0) full = "wu";
                if (full) { strcpy(pin, full); pn = (int)strlen(full); }
                else {
                    /* single_rep_map: v->"yu"+rest, i->"y"+rest, u->"w"+rest, e->unchanged */
                    char f = pin[0];
                    if (f == 'v') { memmove(pin + 2, pin + 1, (size_t)pn); pin[0] = 'y'; pin[1] = 'u'; pn += 1; }
                    else if (f == 'i') pin[0] = 'y';
                    else if (f == 'u') pin[0] = 'w';
                }
            }

            const char *nc, *nv;
            if (!oc_lookup(g, pin, pn, &nc, &nv)) { ia_push(ps, 0); continue; }
            char nvt[16]; snprintf(nvt, sizeof nvt, "%s%c", nv, tone);
            sa_push(ph, arena_dup(g, nc));
            sa_push(ph, arena_dup(g, nvt));
            ia_push(ps, 2);
        }

        free(wcps); free(ini); free(fin); free(itmp); free(ftmp);
    }

    free(merged); free(ar.data); memset(&ar, 0, sizeof ar);
    free(inbuf.data); memset(&inbuf, 0, sizeof inbuf);
    free(in); free(segs); free(fc);
}

/* ---------------- g2p(text): split + per-segment ---------------- */
int gsv_g2p_zh_g2p(gsv_g2p_full *g, const char *text, gsv_g2p_result *out) {
    /* free previous result arrays */
    free(g->keep_ph); free(g->keep_wd); free(g->keep_ps);
    g->keep_ph = g->keep_wd = NULL; g->keep_ps = NULL;

    int n = 0; const char *tp = text; uint32_t c;
    while ((c = gsv_utf8_decode(&tp)) != 0) n++;
    /* Pre-size the arena so it NEVER reallocs during processing: every interned result
     * string pointer stays valid. Total usage is bounded by ~37*n bytes (ini/fin copies +
     * per-syllable phones + word chars); use a wide margin. */
    g->arena.len = 0;
    gsv_buf_reserve(&g->arena, (size_t)n * 256 + 8192);
    uint32_t *cps = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(n + 1));
    int cn = 0; const char *dp = text;
    while ((c = gsv_utf8_decode(&dp)) != 0) cps[cn++] = c;

    strarr ph = {0}; strarr wd = {0}; intarr ps = {0};
    int seg_start = 0;
    for (int i = 0; i < cn; i++) {
        if (is_pause_cp(cps[i])) {
            if (i + 1 > seg_start) process_segment(g, cps, seg_start, i + 1, &ph, &wd, &ps);
            seg_start = i + 1;
            while (seg_start < cn && (cps[seg_start] == 0x20 || cps[seg_start] == 0x09 ||
                   cps[seg_start] == 0x0A || cps[seg_start] == 0x0D)) seg_start++;
        }
    }
    if (seg_start < cn) process_segment(g, cps, seg_start, cn, &ph, &wd, &ps);

    out->n_phones = ph.n; out->phones = ph.a;
    out->n_words  = wd.n; out->words  = wd.a;
    out->n_phs    = ps.n; out->phs    = ps.a;
    g->keep_ph = ph.a; g->keep_wd = wd.a; g->keep_ps = ps.a;
    free(cps);
    return 0;
}

/* ---------------- __init__.py post-processing ---------------- */
static int sym_is_valid(const char *s) {
    int lo = 0, hi = GSV_N_SYMBOLS - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(gsv_symbols[mid], s);
        if (cmp == 0) return 1;
        if (cmp < 0) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

void gsv_g2p_postprocess(gsv_g2p_full *g, gsv_g2p_result *r) {
    (void)g;
    /* 1. pause remap */
    for (int i = 0; i < r->n_phones; i++) {
        const char *q = r->phones[i];
        uint32_t cp = gsv_utf8_decode(&q);
        if (cp == 0) continue;
        const char *sym = pause_symbol_cp(cp);
        if (sym && strcmp(r->phones[i], sym) != 0) r->phones[i] = (char *)sym;
    }
    /* 2. UNK for symbols not in the set */
    for (int i = 0; i < r->n_phones; i++)
        if (!sym_is_valid(r->phones[i])) r->phones[i] = (char *)"UNK";

    /* 3. dedup consecutive punctuation indices */
    int np = r->n_phones > 0 ? r->n_phones : 1;
    int *dup = (int *)malloc(sizeof(int) * (size_t)np);
    int ndup = 0;
    for (int i = 1; i < r->n_phones; i++) {
        if (strcmp(r->phones[i], r->phones[i - 1]) == 0) {
            for (int k = 0; k < GSV_N_PUNCTUATION; k++) {
                if (strcmp(r->phones[i], gsv_punctuation[k]) == 0) { dup[ndup++] = i; break; }
            }
        }
    }

    /* 4. remove UNK/dup phones, adjusting word2ph in reverse */
    int ph_idx = r->n_phones;
    for (int w = r->n_phs - 1; w >= 0; w--) {
        int del = 0;
        for (int k = 0; k < r->phs[w]; k++) {
            ph_idx--;
            int isdup = 0;
            for (int d = 0; d < ndup; d++) if (dup[d] == ph_idx) { isdup = 1; break; }
            if (strcmp(r->phones[ph_idx], "UNK") == 0 || isdup) {
                del++;
                memmove(&r->phones[ph_idx], &r->phones[ph_idx + 1], (size_t)(r->n_phones - ph_idx - 1) * sizeof(char *));
                r->n_phones--;
            }
        }
        r->phs[w] -= del;
        if (r->phs[w] == 0) {
            memmove(&r->phs[w], &r->phs[w + 1], (size_t)(r->n_phs - w - 1) * sizeof(int));
            if (w < r->n_words) memmove(&r->words[w], &r->words[w + 1], (size_t)(r->n_words - w - 1) * sizeof(char *));
            r->n_phs--; r->n_words--;
        }
    }
    free(dup);
}

/* ---------------- text_normalize (chinese.py) ---------------- */
const char *gsv_g2p_zh_text_normalize(gsv_g2p_full *g, const char *text) {
    gsv_buf nb; memset(&nb, 0, sizeof nb);
    const char *norm = gsv_textnorm_normalize(text, &nb);

    gsv_buf fb; memset(&fb, 0, sizeof fb);
    const char *p = norm; uint32_t cp; char eb[8];
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        if (cp == 0x55EF) { memcpy(eb, "\xE6\x81\xA9", 3); gsv_buf_append(&fb, eb, 3); continue; } /* 嗯(U+55EF)->恩(U+6069) */
        if (cp == 0x5463) { memcpy(eb, "\xE6\xAF\x8D", 3); gsv_buf_append(&fb, eb, 3); continue; } /* 呣->母 */
        if (!((cp >= 0x4E00 && cp <= 0x9FA5) || is_pause_cp(cp))) continue;
        int en = gsv_utf8_encode(cp, eb);
        gsv_buf_append(&fb, eb, (size_t)en);
    }

    free(g->normbuf.data); memset(&g->normbuf, 0, sizeof g->normbuf);
    gsv_buf_append(&g->normbuf, fb.data, fb.len);
    gsv_buf_intern(&g->normbuf, "", 0);
    free(nb.data); free(fb.data);
    return g->normbuf.data;
}

/* ---------------- dot-collapse: \.{3,}|。{3,} -> … ---------------- */
static const char *dot_collapse(const char *text, gsv_buf *out) {
    out->len = 0;
    const char *p = text; uint32_t cp; char eb[8];
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        if (cp == 0x2E || cp == 0x3002) {
            uint32_t run = cp; int cnt = 1; const char *q = p;
            /* Count the run WITHOUT over-consuming: gsv_utf8_decode advances q as a
             * side effect, so restore q to the first non-run char before stopping. */
            for (;;) {
                const char *save = q;
                uint32_t c2 = gsv_utf8_decode(&q);
                if (c2 != run) { q = save; break; }
                cnt++;
            }
            if (cnt >= 3) { memcpy(eb, "\xE2\x80\xA6", 3); gsv_buf_append(out, eb, 3); p = q; continue; }
        }
        int en = gsv_utf8_encode(cp, eb);
        gsv_buf_append(out, eb, (size_t)en);
    }
    gsv_buf_intern(out, "", 0);
    return out->data;
}

/* ---------------- text_to_phonemes(text,"zh") ---------------- */
int gsv_g2p_full_to_phonemes(gsv_g2p_full *g, const char *text, gsv_g2p_result *out) {
    gsv_buf dc; memset(&dc, 0, sizeof dc);
    const char *collapsed = dot_collapse(text, &dc);
    const char *norm = gsv_g2p_zh_text_normalize(g, collapsed);
    int rc = gsv_g2p_zh_g2p(g, norm, out);
    if (rc == 0) gsv_g2p_postprocess(g, out);
    free(dc.data); memset(&dc, 0, sizeof dc);
    return rc;
}

/* ---------------- load / free ---------------- */
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = 0; fclose(f);
    if (out_len) *out_len = rd;
    return buf;
}

int gsv_g2p_full_load(gsv_g2p_full *g, const char *dir) {
    memset(g, 0, sizeof *g);
    if (gsv_jieba_load(&g->jb, dir) != 0) return -1;
    if (gsv_g2p_zh_load(&g->py, dir) != 0) return -2;

    char path[1024]; snprintf(path, sizeof path, "%s\\opencpop-strict.txt", dir);
    size_t flen = 0; char *data = read_file(path, &flen);
    if (!data) { snprintf(path, sizeof path, "%s/opencpop-strict.txt", dir); data = read_file(path, &flen); }
    if (!data) return -3;

    g->oc_key = (char **)malloc(sizeof(char *) * 512);
    g->oc_klen = (uint32_t *)malloc(sizeof(uint32_t) * 512);
    g->oc_c = (char **)malloc(sizeof(char *) * 512);
    g->oc_v = (char **)malloc(sizeof(char *) * 512);
    g->n_oc = 0;

    /* Pre-size intern so it NEVER reallocs during the parse loop: every stored
     * oc_key/oc_c/oc_v pointer stays valid (a realloc would dangle earlier ones). */
    gsv_buf_reserve(&g->intern, flen * 2 + 64);

    char *line = data;
    while (*line) {
        char *nl = strchr(line, '\n');
        size_t lmax = nl ? (size_t)(nl - line) : strlen(line);
        if (lmax && line[lmax - 1] == '\r') lmax--;
        char *tab = memchr(line, '\t', lmax);
        if (tab) {
            size_t klen = (size_t)(tab - line);
            const char *val = tab + 1;
            while (*val == ' ') val++;
            /* bound the value to THIS line (val points into the file buffer, so strlen would run to EOF) */
            size_t vlen = (size_t)((line + lmax) - val);
            while (vlen && (val[vlen - 1] == '\r' || val[vlen - 1] == '\n' || val[vlen - 1] == ' ')) vlen--;
            const char *sp = memchr(val, ' ', vlen);
            if (sp) {
                size_t cl = (size_t)(sp - val), vl = vlen - cl - 1;
                int i = g->n_oc++;
                g->oc_key[i] = gsv_buf_intern(&g->intern, line, klen);
                g->oc_klen[i] = (uint32_t)klen;
                g->oc_c[i] = gsv_buf_intern(&g->intern, val, cl);
                g->oc_v[i] = gsv_buf_intern(&g->intern, sp + 1, vl);
            }
        }
        if (!nl) break;
        line = nl + 1;
    }
    free(data);
    return 0;
}

void gsv_g2p_full_free(gsv_g2p_full *g) {
    free(g->keep_ph); free(g->keep_wd); free(g->keep_ps);
    gsv_jieba_free(&g->jb);
    gsv_g2p_zh_free(&g->py);
    free(g->oc_key); free(g->oc_klen); free(g->oc_c); free(g->oc_v);
    free(g->intern.data); memset(&g->intern, 0, sizeof g->intern);
    free(g->normbuf.data); memset(&g->normbuf, 0, sizeof g->normbuf);
    free(g->arena.data); memset(&g->arena, 0, sizeof g->arena);
}
