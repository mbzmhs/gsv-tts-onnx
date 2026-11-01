/* wseg.c - C port of `wordsegment` (Norvig DP segmentation). Exact-match port:
 * same stupid-backoff scores, same log10 sums, first-max tie-break, 250-char
 * chunking with 5-word carry. Per-call memo; entries referenced by stable dense
 * index (not pointer) so memo growth never invalidates a stored suffix link. */
#include "wseg.h"
#include "en_util.h"
#include <math.h>

#define WS_TOTAL 1024908267229.0
#define WS_LIMIT 24
#define WS_CHUNK 250

/* float(10**k) for k=0..24 (Python-correctly-rounded doubles). */
static const double POW10[25] = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24
};

typedef struct mem_ent {
    uint16_t pos;     /* suffix start in base */
    int16_t  ps, pe;  /* prev slice [ps,pe); ps==-1 => "<s>" */
    double   score;
    int16_t  ws, we;  /* winning first word slice [ws,we) */
    int      next;    /* dense index of suffix entry, -1 = terminal */
} mem_ent;

typedef struct {
    char     *base;   int blen;
    mem_ent  *ent;    int ent_cap, ent_n;   /* dense storage */
    uint32_t *slot;   int slot_cap, slot_n; /* open-addr: value = ent index+1, 0=empty */
} ws_call;

/* ---- scoring ---- */
static double score_uni(const gsv_wseg *w, const char *word, int len) {
    uint32_t i = gsv_strmap_get(&w->uni, word, len);
    if (i != UINT32_MAX && i < (uint32_t)w->n_uni) return w->unival[i] / WS_TOTAL;
    return 10.0 / (WS_TOTAL * POW10[len > 24 ? 24 : len]);
}

static double score_prev(const gsv_wseg *w, const char *word, int len,
                         const char *prev, int plen) {
    if (plen < 0) return score_uni(w, word, len);          /* "<s>" */
    /* bigram = prev + " " + word ; requires prev in unigrams too */
    uint32_t pi = gsv_strmap_get(&w->uni, prev, plen);
    if (pi != UINT32_MAX && pi < (uint32_t)w->n_uni) {
        /* build bigram key into a small stack buffer */
        char bg[128];
        int bl = plen + 1 + len;
        if (bl <= (int)sizeof bg) {
            memcpy(bg, prev, (size_t)plen); bg[plen] = ' ';
            memcpy(bg + plen + 1, word, (size_t)len);
            uint32_t bi = gsv_strmap_get(&w->bi, bg, bl);
            if (bi != UINT32_MAX && bi < (uint32_t)w->n_bi)
                return w->bival[bi] / WS_TOTAL / score_uni(w, prev, plen);
        }
    }
    return score_uni(w, word, len);
}

/* ---- memo table (dense entries + open-addressed key->index hash) ---- */
static uint64_t keyhash(uint16_t pos, int16_t ps, int16_t pe) {
    uint64_t h = 14695981039346656037ULL;
    const unsigned char *f = (const unsigned char *)&pos;
    for (int i = 0; i < 2; i++) { h ^= f[i]; h *= 1099511628211ULL; }
    f = (const unsigned char *)&ps;
    for (int i = 0; i < 2; i++) { h ^= f[i]; h *= 1099511628211ULL; }
    f = (const unsigned char *)&pe;
    for (int i = 0; i < 2; i++) { h ^= f[i]; h *= 1099511628211ULL; }
    return h;
}

static int memo_find(ws_call *c, uint16_t pos, int16_t ps, int16_t pe) {
    if (c->slot_cap == 0) return -1;
    uint64_t h = keyhash(pos, ps, pe);
    size_t mask = (size_t)c->slot_cap - 1;
    size_t i = (size_t)(uint32_t)h & mask;
    for (;;) {
        uint32_t v = c->slot[i];
        if (v == 0) return -1;
        mem_ent *e = &c->ent[v - 1];
        if (e->pos == pos && e->ps == ps && e->pe == pe) return (int)(v - 1);
        i = (i + 1) & mask;
    }
}

static int memo_insert(ws_call *c, uint16_t pos, int16_t ps, int16_t pe) {
    if (c->slot_cap == 0 || c->slot_n * 10 >= c->slot_cap * 7) {
        int nc = c->slot_cap ? c->slot_cap * 2 : 4096;
        uint32_t *ns = (uint32_t *)calloc((size_t)nc, 4);
        if (!ns) return -1;
        for (int i = 0; i < c->slot_cap; i++) {
            uint32_t v = c->slot[i];
            if (v == 0) continue;
            mem_ent *e = &c->ent[v - 1];
            uint64_t h = keyhash(e->pos, e->ps, e->pe);
            size_t mask = (size_t)nc - 1, j = (size_t)(uint32_t)h & mask;
            while (ns[j] != 0) j = (j + 1) & mask;
            ns[j] = v;
        }
        free(c->slot); c->slot = ns; c->slot_cap = nc;
    }
    if (c->ent_n >= c->ent_cap) {
        int nc = c->ent_cap ? c->ent_cap * 2 : 4096;
        mem_ent *ne = (mem_ent *)calloc((size_t)nc, sizeof(mem_ent));
        if (!ne) return -1;
        memcpy(ne, c->ent, (size_t)c->ent_n * sizeof(mem_ent));
        free(c->ent); c->ent = ne; c->ent_cap = nc;
    }
    int idx = c->ent_n++;
    mem_ent *e = &c->ent[idx];
    e->pos = pos; e->ps = ps; e->pe = pe; e->score = 0.0; e->ws = -1; e->we = -1; e->next = -1;
    uint64_t h = keyhash(pos, ps, pe);
    size_t mask = (size_t)c->slot_cap - 1;
    size_t i = (size_t)(uint32_t)h & mask;
    while (c->slot[i] != 0) {
        mem_ent *o = &c->ent[c->slot[i] - 1];
        if (o->pos == pos && o->ps == ps && o->pe == pe) { c->ent_n--; return (int)(c->slot[i] - 1); }
        i = (i + 1) & mask;
    }
    c->slot[i] = (uint32_t)(idx + 1); c->slot_n++;
    return idx;
}

/* ---- DP search over base B[0..blen): entry for (pos, prev slice) ---- */
static int ws_search(const gsv_wseg *w, ws_call *c, int pos, int ps, int pe) {
    int n = c->blen;
    if (pos == n) return -1;                    /* terminal: score 0, no words */
    int hit = memo_find(c, (uint16_t)pos, (int16_t)ps, (int16_t)pe);
    if (hit >= 0) return hit;

    int maxplen = n - pos; if (maxplen > WS_LIMIT) maxplen = WS_LIMIT;
    double best = -1e300;
    int bws = -1, bwe = -1, bnext = -1;
    for (int plen = 1; plen <= maxplen; plen++) {
        const char *prefix = c->base + pos;
        double sc = score_prev(w, prefix, plen, ps < 0 ? NULL : c->base + ps, ps < 0 ? -1 : pe - ps);
        double pscore = log10(sc);
        int suf = ws_search(w, c, pos + plen, pos, pos + plen);
        double ssc = (suf >= 0) ? c->ent[suf].score : 0.0;
        double tot = pscore + ssc;
        /* Python max() tie-break: equal score -> larger word-list wins. Both
         * candidate prefixes start at base+pos, so the longer prefix (larger
         * plen) is lexicographically larger. Ascending plen => >= picks it. */
        if (tot >= best) { best = tot; bws = pos; bwe = pos + plen; bnext = suf; }
    }
    int e = memo_insert(c, (uint16_t)pos, (int16_t)ps, (int16_t)pe);
    if (e < 0) return -1;
    c->ent[e].score = best; c->ent[e].ws = (int16_t)bws; c->ent[e].we = (int16_t)bwe; c->ent[e].next = bnext;
    return e;
}

/* ---- public API ---- */
int gsv_wseg_load(gsv_wseg *w, const char *dir) {
    memset(w, 0, sizeof *w);
    size_t len = 0;
    char *buf = gsv_en_read_file(dir, "wseg_uni.bin", &len);
    if (!buf || len < 8) { free(buf); return -1; }
    if (rd_u32(buf) != 0x31555347) { free(buf); return -2; }
    int n = (int)rd_u32(buf + 4);
    const char *p = buf + 8;
    w->n_uni = n;
    w->unival = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!w->unival) return -3;
    if (gsv_strmap_init(&w->uni, n) != 0) return -4;
    for (int i = 0; i < n; i++) {
        int wl = rd_u16(p); p += 2;
        if ((size_t)(p - buf) + (size_t)wl + 8 > len) return -5;
        w->unival[i] = rd_f64(p + wl);
        p += wl + 8;
        if (gsv_strmap_put(&w->uni, p - wl - 8, wl, (uint32_t)i) != 0) return -6;
    }
    free(buf);

    buf = gsv_en_read_file(dir, "wseg_bi.bin", &len);
    if (!buf || len < 8) { free(buf); return -7; }
    if (rd_u32(buf) != 0x31425347) { free(buf); return -8; }
    n = (int)rd_u32(buf + 4);
    p = buf + 8;
    w->n_bi = n;
    w->bival = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!w->bival) return -9;
    if (gsv_strmap_init(&w->bi, n) != 0) return -10;
    for (int i = 0; i < n; i++) {
        int wl = rd_u16(p); p += 2;
        if ((size_t)(p - buf) + (size_t)wl + 8 > len) return -11;
        w->bival[i] = rd_f64(p + wl);
        p += wl + 8;
        if (gsv_strmap_put(&w->bi, p - wl - 8, wl, (uint32_t)i) != 0) return -12;
    }
    free(buf);
    return 0;
}

void gsv_wseg_free(gsv_wseg *w) {
    gsv_strmap_free(&w->uni); gsv_strmap_free(&w->bi);
    free(w->unival); free(w->bival);
    memset(w, 0, sizeof *w);
}

/* file-static scratch (single-threaded TTS pipeline) */
static ws_call s_call;
static char   *s_words = NULL;      /* yielded word bytes */
static size_t  s_words_len = 0, s_words_cap = 0;
static const char **s_out = NULL;   /* result pointer array */
static int     s_out_n = 0, s_out_cap = 0;

static void yield_word(const char *s, int len) {
    if (s_words_len + (size_t)len + 1 > s_words_cap) {
        size_t nc = s_words_cap ? s_words_cap : 256;
        while (nc < s_words_len + (size_t)len + 1) nc *= 2;
        char *nd = (char *)realloc(s_words, nc); if (!nd) return;
        s_words = nd; s_words_cap = nc;
    }
    memcpy(s_words + s_words_len, s, (size_t)len);
    s_words[s_words_len + (size_t)len] = 0;      /* NUL-terminate this word */
    if (s_out_n == s_out_cap) {
        s_out_cap = s_out_cap ? s_out_cap * 2 : 64;
        const char **nd = (const char **)realloc(s_out, (size_t)s_out_cap * sizeof(char *));
        if (!nd) return;
        s_out = nd;
    }
    s_out[s_out_n++] = s_words + s_words_len;
    s_words_len += (size_t)len + 1;              /* skip word AND its NUL byte */
}

/* extract chain of a top entry into yield_word, skipping last `skip` words */
static void emit_chain(int top, int skip) {
    if (top < 0) return;
    int cnt = 0;
    for (int t = top; t >= 0; t = s_call.ent[t].next) cnt++;
    if (cnt < skip) skip = cnt;
    int idx = 0;
    for (int t = top; t >= 0; t = s_call.ent[t].next, idx++) {
        if (idx >= cnt - skip) break;
        yield_word(s_call.base + s_call.ent[t].ws, s_call.ent[t].we - s_call.ent[t].ws);
    }
}

const char **gsv_wseg_segment(const gsv_wseg *w, const char *text, int *n_words) {
    /* clean: lowercase, keep [a-z0-9] */
    s_call.blen = 0;
    if ((size_t)(s_call.base ? strlen(s_call.base) : 0) < strlen(text) + 1) {
        size_t nc = strlen(text) + 1;
        char *nd = (char *)realloc(s_call.base, nc); if (!nd) return NULL;
        s_call.base = nd;
    }
    const char *q = text;
    for (unsigned char c = 0; (c = (unsigned char)*q) != 0; q++) {
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
            s_call.base[s_call.blen++] = (char)c;
    }
    s_call.base[s_call.blen] = 0;

    /* reset per-call scratch */
    s_out_n = 0;
    s_words_len = 0;
    free(s_call.ent); s_call.ent = NULL; s_call.ent_cap = 0; s_call.ent_n = 0;
    free(s_call.slot); s_call.slot = NULL; s_call.slot_cap = 0; s_call.slot_n = 0;

    char prefix[512]; size_t plen = 0;
    prefix[0] = 0;

    int total = s_call.blen;
    for (int off = 0; off < total; off += WS_CHUNK) {
        int clen = total - off; if (clen > WS_CHUNK) clen = WS_CHUNK;
        /* base = prefix + chunk */
        size_t need = plen + (size_t)clen + 1;
        char *nb = (char *)malloc(need);
        if (!nb) return NULL;
        memcpy(nb, prefix, plen);
        memcpy(nb + plen, s_call.base + off, (size_t)clen);
        nb[plen + clen] = 0;
        free(s_call.base);
        s_call.base = nb; s_call.blen = (int)(plen + clen);

        /* fresh memo per chunk */
        free(s_call.ent); s_call.ent = NULL; s_call.ent_cap = 0; s_call.ent_n = 0;
        free(s_call.slot); s_call.slot = NULL; s_call.slot_cap = 0; s_call.slot_n = 0;

        int top = ws_search(w, &s_call, 0, -1, -1);
        emit_chain(top, 5);

        /* new prefix = last 5 words of the full segmentation */
        int cnt = 0;
        for (int t = top; t >= 0; t = s_call.ent[t].next) cnt++;
        if (cnt < 5) cnt = 5;
        size_t np = 0;
        int idx = 0;
        for (int t = top; t >= 0; t = s_call.ent[t].next, idx++) {
            if (idx >= cnt - 5) {
                if (np + (size_t)(s_call.ent[t].we - s_call.ent[t].ws) + 1 <= sizeof prefix) {
                    memcpy(prefix + np, s_call.base + s_call.ent[t].ws, (size_t)(s_call.ent[t].we - s_call.ent[t].ws));
                    np += (size_t)(s_call.ent[t].we - s_call.ent[t].ws);
                }
            }
        }
        prefix[np] = 0; plen = np;
    }

    if (plen > 0) {
        size_t need = plen + 1;
        char *nb = (char *)malloc(need);
        if (!nb) return NULL;
        memcpy(nb, prefix, plen); nb[plen] = 0;
        free(s_call.base);
        s_call.base = nb; s_call.blen = (int)plen;
        free(s_call.ent); s_call.ent = NULL; s_call.ent_cap = 0; s_call.ent_n = 0;
        free(s_call.slot); s_call.slot = NULL; s_call.slot_cap = 0; s_call.slot_n = 0;
        int top = ws_search(w, &s_call, 0, -1, -1);
        emit_chain(top, 0);
    }

    if (n_words) *n_words = s_out_n;
    return s_out;
}
