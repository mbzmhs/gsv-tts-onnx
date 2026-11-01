/* g2p_en.c - C port of the English G2P orchestration (english.py + __init__.py "en").
 * Composes verified components: en_textnorm, tweet_tokenizer, pos_tag_en,
 * en_dicts (cmu/namedict), wseg, gru_predict, homographs_en. */
#include "g2p_en.h"
#include "en_textnorm.h"
#include "tweet_tokenizer.h"
#include "en_phonemes.h"
#include "homographs_en.h"
#include "symbols.h"

#define HOMO_N ((int)(sizeof(gsv_en_homographs) / sizeof(gsv_en_homographs[0])))

/* ---- symbol membership (Symbols.symbols / Symbols.punctuation) ---- */
static int en_is_symbol(const char *s) {
    for (int i = 0; i < GSV_N_SYMBOLS; i++) if (!strcmp(gsv_symbols[i], s)) return 1;
    return 0;
}
static int en_is_punct(const char *s) {
    for (int i = 0; i < GSV_N_PUNCTUATION; i++) if (!strcmp(gsv_punctuation[i], s)) return 1;
    return 0;
}

/* ---- Pause.pause_map ---- */
static const struct { const char *k, *v; } PAUSE_MAP[] = {
    {"!", "!"},   {"！", "!"},   {"?", "?"},   {"？", "?"},
    {"…", "…"},   {"~", "…"},   {".", "."},   {"。", "."},
    {",", ","},   {"，", ","},   {":", ","},   {"：", ","},
    {";", ","},   {"；", ","},   {"、", ","},   {"・", ","}
};
#define PAUSE_MAP_N ((int)(sizeof(PAUSE_MAP) / sizeof(PAUSE_MAP[0])))

static const char *pause_map_get(const char *s) {
    for (int i = 0; i < PAUSE_MAP_N; i++) if (!strcmp(PAUSE_MAP[i].k, s)) return PAUSE_MAP[i].v;
    return NULL;
}

/* is codepoint a single-codepoint pause key? */
static int is_pause_cp(uint32_t cp) {
    for (int i = 0; i < PAUSE_MAP_N; i++) {
        const char *k = PAUSE_MAP[i].k; const char *p = k;
        uint32_t kc = gsv_utf8_decode(&p);
        if (kc == cp && *p == 0) return 1;
    }
    return 0;
}

static int is_ws_cp(uint32_t cp) {
    switch (cp) { case 0x20: case 0x09: case 0x0A: case 0x0D: case 0x0C: case 0x0B: return 1; }
    return 0;
}

/* ---- small string helpers ---- */
static void lower_copy(char *dst, const char *src) {
    for (const unsigned char *p = (const unsigned char *)src;; p++, dst++) {
        unsigned char c = *p;
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        *dst = (char)c;
        if (!*p) break;
    }
}
static int has_lower_alpha(const char *w) {
    for (const unsigned char *p = (const unsigned char *)w; *p; p++)
        if (*p >= 'a' && *p <= 'z') return 1;
    return 0;
}
/* Python str.istitle() semantics over the (ASCII) token. */
static int en_istitle(const char *s) {
    int prev_cased = 0, any_cased = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        int isupper = (c >= 'A' && c <= 'Z'), islower = (c >= 'a' && c <= 'z');
        int cased = isupper || islower;
        if (cased) {
            any_cased = 1;
            if (!prev_cased && !isupper) return 0;
            if (prev_cased && isupper) return 0;
        }
        prev_cased = cased;
    }
    return any_cased;
}

/* ---- temporary phoneme list ---- */
typedef struct { const char **a; int n, cap; } en_pl;
static void pl_init(en_pl *p) { p->a = NULL; p->n = 0; p->cap = 0; }
static int pl_push(en_pl *p, const char *s) {
    if (p->n == p->cap) {
        int nc = p->cap ? p->cap * 2 : 8;
        const char **nd = (const char **)realloc(p->a, (size_t)nc * sizeof(*nd));
        if (!nd) return -1;
        p->a = nd; p->cap = nc;
    }
    p->a[p->n++] = s;
    return 0;
}
static void pl_extend(en_pl *d, const en_pl *s) { for (int i = 0; i < s->n; i++) pl_push(d, s->a[i]); }

/* number of phonemes in a pron excluding <pad>/UW/</s>/<s> (== word2ph["ph"]). */
static int pron_phcount(const en_pl *pl) {
    int c = 0;
    for (int i = 0; i < pl->n; i++) {
        const char *s = pl->a[i];
        if (strcmp(s, "<pad>") && strcmp(s, "UW") && strcmp(s, "</s>") && strcmp(s, "<s>")) c++;
    }
    return c;
}

static int pl_from_dict(gsv_g2p_en_full *g, gsv_en_dict *dict, const char *word, en_pl *out) {
    (void)g;
    int nn = 0;
    const uint8_t *idx = gsv_en_dict_get(dict, word, &nn);
    if (!idx) return 0;
    for (int i = 0; i < nn; i++) pl_push(out, gsv_en_phonemes[idx[i]]);
    return 1;
}

/* split a space-separated phoneme string ("R IY1 D") into out, interning tokens. */
static void pl_from_pronstr(gsv_g2p_en_full *g, const char *s, en_pl *out) {
    const char *p = s;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *st = p;
        while (*p && *p != ' ') p++;
        int len = (int)(p - st);
        char *cp = gsv_buf_intern(&g->arena, st, (size_t)len);
        if (cp) pl_push(out, cp);
    }
}

static const gsv_en_homo *find_homograph(const char *word) {
    for (int i = 0; i < HOMO_N; i++) if (!strcmp(gsv_en_homographs[i].word, word)) return &gsv_en_homographs[i];
    return NULL;
}

/* ---- english.py qryword(o_word) ---- */
static void gsv_qryword(gsv_g2p_en_full *g, const char *o_word, en_pl *out) {
    char wbuf[1024];
    lower_copy(wbuf, o_word);
    int wl = (int)strlen(wbuf);

    /* 1. CMU dict (len>1) */
    if (wl > 1 && pl_from_dict(g, &g->cmu, wbuf, out)) return;

    /* 2. namedict when o_word is title-cased */
    if (en_istitle(o_word) && pl_from_dict(g, &g->namedict, wbuf, out)) return;

    /* 3. len<=3: read letter-by-letter */
    if (wl <= 3) {
        for (int i = 0; i < wl; i++) {
            unsigned char c = (unsigned char)wbuf[i];
            if (c == 'a') pl_push(out, "EY1");
            else if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                char one[2] = { (char)c, 0 };
                char *cp = gsv_buf_intern(&g->arena, one, 1);
                pl_push(out, cp ? cp : one);
            } else {
                char one[2] = { (char)c, 0 };
                en_pl t; pl_init(&t);
                if (pl_from_dict(g, &g->cmu, one, &t)) pl_extend(out, &t);
                free(t.a);
            }
        }
        return;
    }

    /* 4. possessive ^([a-z]+)('s)$  (dead after normalization: apostrophes are stripped) */
    if (wl >= 3 && wbuf[wl - 2] == '\'' && wbuf[wl - 1] == 's') {
        int ok = 1;
        for (int i = 0; i < wl - 2; i++) { unsigned char c = (unsigned char)wbuf[i]; if (!((c >= 'a' && c <= 'z'))) { ok = 0; break; } }
        if (ok) {
            char stem[1024];
            memcpy(stem, wbuf, (size_t)(wl - 2)); stem[wl - 2] = 0;
            en_pl sp; pl_init(&sp);
            gsv_qryword(g, stem, &sp);
            pl_extend(out, &sp);
            if (sp.n > 0) {
                const char *last = sp.a[sp.n - 1];
                if (!strcmp(last, "P") || !strcmp(last, "T") || !strcmp(last, "K") ||
                    !strcmp(last, "F") || !strcmp(last, "TH") || !strcmp(last, "HH")) pl_push(out, "S");
                else if (!strcmp(last, "S") || !strcmp(last, "Z") || !strcmp(last, "SH") ||
                         !strcmp(last, "ZH") || !strcmp(last, "CH") || !strcmp(last, "JH")) { pl_push(out, "AH0"); pl_push(out, "Z"); }
                else pl_push(out, "Z");
            }
            free(sp.a);
            return;
        }
    }

    /* 5. wordsegment (compound) or GRU predict */
    int nc = 0;
    const char **comps = gsv_wseg_segment(&g->wseg, wbuf, &nc);
    if (nc == 1) {
        const char *gru_ph[24]; int np = 0;
        gsv_gru_predict(&g->gru, gru_ph, wbuf, &np);
        for (int i = 0; i < np && i < 24; i++) pl_push(out, gru_ph[i]);
        return;
    }
    /* copy comps: recursive segment() calls would overwrite the wseg buffer */
    const char **cc = (const char **)malloc((size_t)(nc > 0 ? nc : 1) * sizeof(char *));
    for (int i = 0; i < nc; i++) { int l = (int)strlen(comps[i]); cc[i] = gsv_buf_intern(&g->arena, comps[i], (size_t)l); }
    for (int i = 0; i < nc; i++) {
        en_pl cp; pl_init(&cp);
        if (cc[i]) gsv_qryword(g, cc[i], &cp);
        pl_extend(out, &cp);
        free(cp.a);
    }
    free(cc);
}

/* ---- english.py text_normalize ---- */
const char *gsv_g2p_en_text_normalize(gsv_g2p_en_full *g, const char *text) {
    gsv_buf tmp; memset(&tmp, 0, sizeof tmp);
    if (gsv_en_textnorm_normalize(text, &tmp) != 0) { free(tmp.data); return ""; }

    gsv_buf *nb = &g->normbuf;
    nb->len = 0;
    const char *p = tmp.data; uint32_t cp; int inws = 0;
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        unsigned char c = (unsigned char)cp;
        int keep;
        if (cp < 0x80)
            keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_ws_cp(cp) || is_pause_cp(cp);
        else
            keep = is_pause_cp(cp);
        if (!keep) continue;
        if (is_ws_cp(cp)) {
            if (inws) continue;
            inws = 1; gsv_buf_append_char(nb, ' ');
        } else {
            inws = 0;
            char enc[4]; int e = gsv_utf8_encode(cp, enc); gsv_buf_append(nb, enc, (size_t)e);
        }
    }
    if (nb->len < nb->cap) gsv_buf_append_char(nb, 0); else { /* ensure NUL */ char z=0; gsv_buf_append(nb,&z,1); }
    free(tmp.data);
    return nb->data;
}

/* ---- english.py g2p(norm_text) ---- */
int gsv_g2p_en_g2p(gsv_g2p_en_full *g, const char *norm_text, gsv_g2p_en_result *out) {
    /* reset per-call state */
    if (g->tok_arena) { free(g->tok_arena); g->tok_arena = NULL; }
    g->keep_ph_n = 0; g->keep_wd_n = 0; g->keep_ps_n = 0;
    g->arena.len = 0;
    size_t tl = strlen(norm_text);
    gsv_buf_reserve(&g->arena, (tl + 64) * 512);

    const char **tokens = NULL; size_t ntok = 0; void *arena = NULL;
    if (gsv_tweet_tokenize(norm_text, tl, &tokens, &ntok, &arena) != 0) return -1;
    g->tok_arena = arena;

    const char **tags = (const char **)malloc((size_t)(ntok ? ntok : 1) * sizeof(char *));
    if (!tags) return -2;
    gsv_pos_tag(&g->pos, tokens, (int)ntok, tags);

    const char **prons = NULL; int npr = 0, cappr = 0;
    char **words = (char **)malloc((size_t)(ntok ? ntok : 1) * sizeof(char *));
    int *phs = (int *)malloc((size_t)(ntok ? ntok : 1) * sizeof(int));
    int nw = 0;

    for (size_t i = 0; i < ntok; i++) {
        const char *o_word = tokens[i];
        const char *pos = tags[i];
        en_pl pron; pl_init(&pron);
        char wbuf[1024]; lower_copy(wbuf, o_word);

        if (!has_lower_alpha(wbuf)) {
            int wl = (int)strlen(wbuf);
            char *cp = gsv_buf_intern(&g->arena, wbuf, (size_t)wl);
            pl_push(&pron, cp ? cp : wbuf);
        } else if (strlen(wbuf) == 1) {
            if (!strcmp(o_word, "A")) pl_push(&pron, "EY1");
            else { en_pl t; pl_init(&t); if (!pl_from_dict(g, &g->cmu, wbuf, &t)) pl_push(&t, wbuf); pl_extend(&pron, &t); free(t.a); }
        } else if (find_homograph(wbuf)) {
            const gsv_en_homo *hm = find_homograph(wbuf);
            const char *pos1 = hm->pos;
            int use1 = 0;
            if (!strncmp(pos, pos1, strlen(pos1))) use1 = 1;
            else if ((int)strlen(pos) < (int)strlen(pos1) && !strncmp(pos, pos1, strlen(pos))) use1 = 1;
            pl_from_pronstr(g, use1 ? hm->pron1 : hm->pron2, &pron);
        } else {
            gsv_qryword(g, o_word, &pron);
        }

        for (int k = 0; k < pron.n; k++) {
            if (npr == cappr) { cappr = cappr ? cappr * 2 : 32; const char **nd = (const char **)realloc(prons, (size_t)cappr * sizeof(char *)); if (!nd) break; prons = nd; }
            prons[npr++] = pron.a[k];
        }
        if (npr == cappr) { cappr = cappr ? cappr * 2 : 32; const char **nd = (const char **)realloc(prons, (size_t)cappr * sizeof(char *)); if (nd) prons = nd; }
        if (npr < cappr) prons[npr++] = " ";

        words[nw] = (char *)o_word;
        phs[nw] = pron_phcount(&pron);
        nw++;
        free(pron.a);
    }

    /* g2p() filter: drop {" ","<pad>","UW","</s>","<s>"}, map <unk>->UNK */
    int nph = 0;
    for (int i = 0; i < npr; i++) {
        const char *p = prons[i];
        if (!strcmp(p, " ") || !strcmp(p, "<pad>") || !strcmp(p, "UW") || !strcmp(p, "</s>") || !strcmp(p, "<s>")) continue;
        if (g->keep_ph_n == g->keep_ph_cap) {
            int nc = g->keep_ph_cap ? g->keep_ph_cap * 2 : 64;
            char **nd = (char **)realloc(g->keep_ph, (size_t)nc * sizeof(char *)); if (!nd) break;
            g->keep_ph = nd; g->keep_ph_cap = nc;
        }
        g->keep_ph[g->keep_ph_n++] = (strcmp(p, "<unk>") ? (char *)p : "UNK");
    }
    nph = g->keep_ph_n;

    for (int i = 0; i < nw; i++) {
        if (g->keep_wd_n == g->keep_wd_cap) { int nc = g->keep_wd_cap ? g->keep_wd_cap * 2 : 64; char **nd = (char **)realloc(g->keep_wd, (size_t)nc * sizeof(char *)); if (!nd) break; g->keep_wd = nd; g->keep_wd_cap = nc; }
        g->keep_wd[g->keep_wd_n++] = words[i];
        if (g->keep_ps_n == g->keep_ps_cap) { int nc = g->keep_ps_cap ? g->keep_ps_cap * 2 : 64; int *nd = (int *)realloc(g->keep_ps, (size_t)nc * sizeof(int)); if (!nd) break; g->keep_ps = nd; g->keep_ps_cap = nc; }
        g->keep_ps[g->keep_ps_n++] = phs[i];
    }

    out->phones = g->keep_ph; out->n_phones = nph;
    out->words = g->keep_wd; out->n_words = nw;
    out->phs = g->keep_ps; out->n_phs = nw;

    free(tags); free(words); free(phs); free(prons);
    return 0;
}

/* ---- __init__.py post-processing (pause remap + UNK-map + dedup + filter) ---- */
static int is_dup(int idx, const int *dup, int nd) { for (int i = 0; i < nd; i++) if (dup[i] == idx) return 1; return 0; }

void gsv_g2p_en_postprocess(gsv_g2p_en_full *g, gsv_g2p_en_result *r) {
    int n = r->n_phones;
    char **np = (char **)malloc((size_t)(n > 0 ? n : 1) * sizeof(char *));
    for (int i = 0; i < n; i++) {
        const char *p = r->phones[i];
        const char *m = pause_map_get(p); if (m) p = m;
        np[i] = en_is_symbol(p) ? (char *)p : "UNK";
    }
    int *dup = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int)); int ndup = 0;
    for (int i = 1; i < n; i++) if (!strcmp(np[i], np[i - 1]) && en_is_punct(np[i])) dup[ndup++] = i;

    char *survive = (char *)malloc((size_t)(n > 0 ? n : 1)); for (int i = 0; i < n; i++) survive[i] = 1;
    int *wcnt = (int *)malloc((size_t)(r->n_words > 0 ? r->n_words : 1) * sizeof(int));
    int ph_idx = n;
    for (int w = r->n_words - 1; w >= 0; w--) {
        int cnt = r->phs[w], del = 0;
        for (int k = 0; k < cnt; k++) {
            ph_idx--;
            if (!strcmp(np[ph_idx], "UNK") || is_dup(ph_idx, dup, ndup)) { survive[ph_idx] = 0; del++; }
        }
        wcnt[w] = cnt - del;
    }

    char **fph = (char **)malloc((size_t)(n > 0 ? n : 1) * sizeof(char *)); int fn = 0;
    for (int i = 0; i < n; i++) if (survive[i]) fph[fn++] = np[i];
    char **fw = (char **)malloc((size_t)(r->n_words > 0 ? r->n_words : 1) * sizeof(char *));
    int *fp = (int *)malloc((size_t)(r->n_words > 0 ? r->n_words : 1) * sizeof(int)); int nfw = 0;
    for (int w = 0; w < r->n_words; w++) if (wcnt[w] > 0) { fw[nfw] = r->words[w]; fp[nfw] = wcnt[w]; nfw++; }

    free(g->keep_ph); g->keep_ph = fph; g->keep_ph_n = fn; g->keep_ph_cap = n > 0 ? n : 1;
    free(g->keep_wd); g->keep_wd = fw; g->keep_wd_n = nfw; g->keep_wd_cap = r->n_words > 0 ? r->n_words : 1;
    free(g->keep_ps); g->keep_ps = fp; g->keep_ps_n = nfw; g->keep_ps_cap = r->n_words > 0 ? r->n_words : 1;

    r->phones = g->keep_ph; r->n_phones = fn;
    r->words = g->keep_wd; r->n_words = nfw;
    r->phs = g->keep_ps; r->n_phs = nfw;

    free(np); free(dup); free(survive); free(wcnt);
}

/* ---- __init__.py text_to_phonemes(text, "en") ---- */
static char *dot_collapse(const char *text) {
    gsv_buf b; memset(&b, 0, sizeof b);
    const char *p = text; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        if (cp == '.' || cp == 0x3002) {
            int bsz = (cp < 0x80) ? 1 : 3;             /* '.'=1, '。'=3 */
            const char *run_start = p - bsz;           /* points at first dot */
            int run = 1; const char *end = p;          /* end = just after last dot */
            const char *q = p; uint32_t c2;
            while ((c2 = gsv_utf8_decode(&q)) != 0 && c2 == cp) { run++; end = q; }
            if (run >= 3) gsv_buf_append_str(&b, "\xE2\x80\xA6");   /* … */
            else gsv_buf_append(&b, run_start, (size_t)run * bsz);
            p = end; continue;                         /* reprocess first non-dot char */
        }
        char enc[4]; int e = gsv_utf8_encode(cp, enc); gsv_buf_append(&b, enc, (size_t)e);
    }
    if (b.len < b.cap) { char z = 0; gsv_buf_append(&b, &z, 1); }
    return b.data;
}

int gsv_g2p_en_to_phonemes(gsv_g2p_en_full *g, const char *text, gsv_g2p_en_result *out) {
    char *collapsed = dot_collapse(text);
    const char *norm = gsv_g2p_en_text_normalize(g, collapsed);
    free(collapsed);
    int rc = gsv_g2p_en_g2p(g, norm, out);
    if (rc != 0) return rc;
    gsv_g2p_en_postprocess(g, out);
    return 0;
}

/* ---- load / free ---- */
int gsv_g2p_en_load(gsv_g2p_en_full *g, const char *dir) {
    memset(g, 0, sizeof *g);
    if (gsv_en_dict_load(&g->cmu, dir, "cmu.bin", 0x4D444331u) != 0) return -1;
    if (gsv_en_dict_load(&g->namedict, dir, "namedict.bin", 0x4D444E31u) != 0) return -2;
    if (gsv_pos_tag_load(&g->pos, dir) != 0) return -3;
    if (gsv_gru_load(&g->gru, dir) != 0) return -4;
    if (gsv_wseg_load(&g->wseg, dir) != 0) return -5;
    return 0;
}

void gsv_g2p_en_free(gsv_g2p_en_full *g) {
    gsv_en_dict_free(&g->cmu); gsv_en_dict_free(&g->namedict);
    gsv_pos_tag_free(&g->pos); gsv_gru_free(&g->gru); gsv_wseg_free(&g->wseg);
    free(g->normbuf.data); free(g->arena.data); free(g->tok_arena);
    free(g->keep_ph); free(g->keep_wd); free(g->keep_ps);
    memset(g, 0, sizeof *g);
}
