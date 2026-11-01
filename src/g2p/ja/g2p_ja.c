/* g2p_ja.c - C port of GPT_SoVITS/G2P/Japanese/japanese.py (OpenJTalk base only).
 * Uses the OpenJTalk 1.11 C frontend (Mecab + NJD + JPCommon) for run_frontend and
 * make_label, then ports pyopenjtalk_g2p_prosody / preprocess_jap / g2p verbatim. */
#include "g2p_ja.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mecab.h"
#include "njd.h"
#include "jpcommon.h"
#include "text2mecab.h"
#include "mecab2njd.h"
#include "njd_set_pronunciation.h"
#include "njd_set_digit.h"
#include "njd_set_accent_phrase.h"
#include "njd_set_accent_type.h"
#include "njd_set_unvoiced_vowel.h"
#include "njd_set_long_vowel.h"
#include "njd2jpcommon.h"

struct gsv_g2p_ja {
    Mecab mecab;
    NJD njd;
    JPCommon jpcommon;
    int loaded;

    /* result string ownership: every interned result token is xstrdup'd and tracked
     * here so it can be freed on the next g2p call or on free(). */
    char **res_strs;
    int res_n, res_cap;
};

/* ---------- small helpers ---------- */

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (!d) return NULL;
    memcpy(d, s, n);
    return d;
}

/* japanese.py _japanese_characters: [A-Za-z\d\u3005\u3040-\u30fa\u30fc-\u30ff
 * \u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d] */
static int is_jp_char(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return 1;
    if (cp >= 'a' && cp <= 'z') return 1;
    if (cp >= '0' && cp <= '9') return 1;
    if (cp == 0x3005) return 1;                       /* \u3005 */
    if (cp >= 0x3040 && cp <= 0x30FA) return 1;       /* hiragana */
    if (cp >= 0x30FC && cp <= 0x30FF) return 1;       /* katakana incl \u30fc-\u30ff */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;       /* CJK unified */
    if (cp >= 0xFF11 && cp <= 0xFF19) return 1;       /* fullwidth digits */
    if (cp >= 0xFF21 && cp <= 0xFF3A) return 1;       /* fullwidth A-Z */
    if (cp >= 0xFF41 && cp <= 0xFF5A) return 1;       /* fullwidth a-z */
    if (cp >= 0xFF66 && cp <= 0xFF9D) return 1;       /* halfwidth katakana */
    return 0;
}

/* growable array of owned strings */
typedef struct { char **a; int n, cap; } strarr;

static void sarr_init(strarr *s) { s->a = NULL; s->n = 0; s->cap = 0; }
static void sarr_push(strarr *s, const char *tok) {
    if (s->n == s->cap) {
        int nc = s->cap ? s->cap * 2 : 8;
        char **na = (char **)realloc(s->a, (size_t)nc * sizeof(char *));
        if (!na) return;
        s->a = na; s->cap = nc;
    }
    s->a[s->n++] = xstrdup(tok);
}
static void sarr_free(strarr *s) {
    for (int i = 0; i < s->n; i++) free(s->a[i]);
    free(s->a);
    sarr_init(s);
}

/* intern a result token into the handle (owned, freed on next call/free). */
static char *res_intern(gsv_g2p_ja *j, const char *s) {
    if (j->res_n == j->res_cap) {
        int nc = j->res_cap ? j->res_cap * 2 : 16;
        char **na = (char **)realloc(j->res_strs, (size_t)nc * sizeof(char *));
        if (!na) return NULL;
        j->res_strs = na; j->res_cap = nc;
    }
    j->res_strs[j->res_n++] = xstrdup(s);
    return j->res_strs[j->res_n - 1];
}

static void res_clear(gsv_g2p_ja *j) {
    for (int i = 0; i < j->res_n; i++) free(j->res_strs[i]);
    j->res_n = 0;
}

/* ---------- label field parsing (mirror python re.search semantics) ---------- */

/* p3 = substring between the first '-' and the following '+'. Returns ptr+len or NULL. */
static const char *label_p3(const char *lab, int *outlen) {
    const char *d = strchr(lab, '-');
    if (!d) return NULL;
    d++;
    const char *p = strchr(d, '+');
    if (!p) return NULL;
    *outlen = (int)(p - d);
    return d;
}

/* /A:([0-9\-]+)\+  : first number in the A: field */
static long a1_of(const char *s) {
    const char *p = strstr(s, "/A:");
    if (!p) return -50;
    p += 3;
    int neg = 0, any = 0;
    long v = 0;
    while (*p && *p != '+') {
        if (*p == '-' && !any) { neg = 1; p++; }
        else if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; any = 1; }
        else break;
    }
    return any ? (neg ? -v : v) : -50;
}

/* \+(\d+)\+ : first "+<digits>+" */
static long a2_of(const char *s) {
    const char *p = s;
    while ((p = strchr(p, '+')) != NULL) {
        const char *d = p + 1;
        if (*d >= '0' && *d <= '9') {
            long v = 0;
            while (*d >= '0' && *d <= '9') { v = v * 10 + (*d - '0'); d++; }
            if (*d == '+') return v;
        }
        p++;
    }
    return -50;
}

/* \+(\d+)/ : first "+<digits>/" */
static long a3_of(const char *s) {
    const char *p = s;
    while ((p = strchr(p, '+')) != NULL) {
        const char *d = p + 1;
        if (*d >= '0' && *d <= '9') {
            long v = 0;
            while (*d >= '0' && *d <= '9') { v = v * 10 + (*d - '0'); d++; }
            if (*d == '/') return v;
        }
        p++;
    }
    return -50;
}

/* /F:(\d+)_ */
static long f1_of(const char *s) {
    const char *p = strstr(s, "/F:");
    if (!p) return -50;
    p += 3;
    long v = 0; int any = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; any = 1; }
    if (!any) return -50;
    return (*p == '_') ? v : -50;
}

/* !(\d+)_ */
static long e3_of(const char *s) {
    const char *p = s;
    while ((p = strchr(p, '!')) != NULL) {
        const char *d = p + 1;
        if (*d >= '0' && *d <= '9') {
            long v = 0; int any = 0;
            while (*d >= '0' && *d <= '9') { v = v * 10 + (*d - '0'); d++; any = 1; }
            if (any && *d == '_') return v;
        }
        p++;
    }
    return -50;
}

/* ---------- OpenJTalk frontend ---------- */

/* Run the full OpenJTalk frontend on `text`. Copies per-node string/pron and the
 * JPCommon labels out as owned (xstrdup) arrays. Returns 0 on success. */
static int run_frontend(gsv_g2p_ja *j, const char *text,
                        char ***out_str, int *out_nstr,
                        char ***out_pron, int *out_npron,
                        char ***out_labels, int *out_nlabels) {
    if (strlen(text) >= 8192) return -1;
    char buff[8192];
    text2mecab(buff, text);
    Mecab_analysis(&j->mecab, buff);
    mecab2njd(&j->njd, Mecab_get_feature(&j->mecab), Mecab_get_size(&j->mecab));
    njd_set_pronunciation(&j->njd);
    njd_set_digit(&j->njd);
    njd_set_accent_phrase(&j->njd);
    njd_set_accent_type(&j->njd);
    njd_set_unvoiced_vowel(&j->njd);
    njd_set_long_vowel(&j->njd);
    njd2jpcommon(&j->jpcommon, &j->njd);
    JPCommon_make_label(&j->jpcommon);

    int n = 0;
    for (NJDNode *nd = j->njd.head; nd; nd = nd->next) n++;
    size_t nn = (size_t)(n > 0 ? n : 1);
    char **strs = (char **)malloc(nn * sizeof(char *));
    char **prons = (char **)malloc(nn * sizeof(char *));
    if (!strs || !prons) { free(strs); free(prons); return -1; }
    int k = 0;
    for (NJDNode *nd = j->njd.head; nd; nd = nd->next, k++) {
        strs[k] = xstrdup(nd->string ? nd->string : "");
        prons[k] = xstrdup(nd->pron ? nd->pron : "");
    }

    int ls = JPCommon_get_label_size(&j->jpcommon);
    char **lf = JPCommon_get_label_feature(&j->jpcommon);
    size_t ln = (size_t)(ls > 0 ? ls : 1);
    char **labels = (char **)malloc(ln * sizeof(char *));
    if (!labels) { for (int i = 0; i < n; i++) { free(strs[i]); free(prons[i]); } free(strs); free(prons); return -1; }
    for (int i = 0; i < ls; i++) labels[i] = xstrdup(lf[i]);

    JPCommon_refresh(&j->jpcommon);
    NJD_refresh(&j->njd);
    Mecab_refresh(&j->mecab);

    *out_str = strs; *out_nstr = n;
    *out_pron = prons; *out_npron = n;
    *out_labels = labels; *out_nlabels = ls;
    return 0;
}

static void free_frontend_out(char **strs, int nstr, char **prons, int nprom,
                              char **labels, int nlab) {
    for (int i = 0; i < nstr; i++) { free(strs[i]); free(prons[i]); }
    free(strs); free(prons);
    for (int i = 0; i < nlab; i++) free(labels[i]);
    free(labels);
    (void)nprom;
}

/* Count base phones for a single katakana pronunciation word: run the frontend on it
 * and count labels whose p3 is neither "sil" nor "pau". Matches len(g2p(pron).split()). */
static int base_phone_count(gsv_g2p_ja *j, const char *pron) {
    if (!pron || !pron[0]) return 0;
    char **strs = NULL; int nstr = 0;
    char **prons = NULL; int nprom = 0;
    char **labels = NULL; int nlab = 0;
    if (run_frontend(j, pron, &strs, &nstr, &prons, &nprom, &labels, &nlab) != 0) return 0;
    int cnt = 0;
    for (int i = 0; i < nlab; i++) {
        int len; const char *p3 = label_p3(labels[i], &len);
        if (!p3) continue;
        if ((len == 3 && strncmp(p3, "sil", 3) == 0)) continue;
        if ((len == 3 && strncmp(p3, "pau", 3) == 0)) continue;
        cnt++;
    }
    free_frontend_out(strs, nstr, prons, nprom, labels, nlab);
    return cnt;
}

/* ---------- pyopenjtalk_g2p_prosody ---------- */

/* Port of JapaneseG2P.pyopenjtalk_g2p_prosody(text, word2ph). Appends the sentence's
 * ph[1:-1] phones to `phones` and its word2ph entries to `words`/`phs`. */
static int g2p_prosody(gsv_g2p_ja *j, const char *text,
                       strarr *phones, strarr *words, int **phs_out, int *phs_n_out,
                       int *phs_cap_out) {
    char **fstr = NULL; int nfstr = 0;
    char **fpron = NULL; int nfpron = 0;
    char **labels = NULL; int N = 0;
    if (run_frontend(j, text, &fstr, &nfstr, &fpron, &nfpron, &labels, &N) != 0) return -1;

    /* expected_base_counts */
    int *ebc = (int *)malloc(((size_t)(nfpron > 0 ? nfpron : 1)) * sizeof(int));
    for (int i = 0; i < nfpron; i++) {
        if (strcmp(fpron[i], "IDLE") == 0) ebc[i] = 0;
        else ebc[i] = base_phone_count(j, fpron[i]);
    }

    int *node_pc = (int *)calloc((size_t)(nfpron > 0 ? nfpron : 1), sizeof(int));

    /* local phones list for this sentence */
    strarr local; sarr_init(&local);

    int node_idx = 0;
    long current_base_consumed = 0;
    while (node_idx < nfpron - 1 && ebc[node_idx] == 0) node_idx++;

    for (int n = 0; n < N; n++) {
        const char *lab_curr = labels[n];
        int plen;
        const char *p3ptr = label_p3(lab_curr, &plen);
        if (!p3ptr) continue;
        char p3[16];
        if (plen >= (int)(sizeof(p3))) plen = (int)(sizeof(p3)) - 1;
        memcpy(p3, p3ptr, (size_t)plen);
        p3[plen] = '\0';

        /* drop_unvoiced_vowels: AEIOU -> aeiou */
        if (plen == 1 && (p3[0] == 'A' || p3[0] == 'E' || p3[0] == 'I' || p3[0] == 'O' || p3[0] == 'U'))
            p3[0] = (char)(p3[0] + 32);

        const char *prosody_mark = NULL;
        int is_sil_pau = (strcmp(p3, "sil") == 0) || (strcmp(p3, "pau") == 0);
        if (!is_sil_pau) {
            long a1 = a1_of(lab_curr);
            long a2 = a2_of(lab_curr);
            long a3 = a3_of(lab_curr);
            long f1 = f1_of(lab_curr);
            long a2_next = (n + 1 < N) ? a2_of(labels[n + 1]) : -1;

            int p3_in_aeiouNcl = 0;
            if (plen == 1) {
                const char *set = "aeiouAEIOUNcl"; /* exact python set */
                for (const char *s = set; *s; s++)
                    if (*s == p3[0]) { p3_in_aeiouNcl = 1; break; }
            }

            if (a3 == 1 && a2_next == 1 && p3_in_aeiouNcl) {
                prosody_mark = "#";
            } else if (a1 == 0 && a2_next == a2 + 1 && a2 != f1) {
                prosody_mark = "]";
            } else if (a2 == 1 && a2_next == 2) {
                prosody_mark = "[";
            }

            if (prosody_mark) node_pc[node_idx] += 1;
        }

        const char *res_p = NULL;
        int is_sentence_boundary_sil = 0;
        if (strcmp(p3, "sil") == 0) {
            if (n == 0) { res_p = "^"; is_sentence_boundary_sil = 1; }
            else if (n == N - 1) {
                long e3 = e3_of(lab_curr);
                res_p = (e3 == 0) ? "$" : "?";
                is_sentence_boundary_sil = 1;
            } else {
                res_p = "_";
            }
        } else if (strcmp(p3, "pau") == 0) {
            res_p = "_";
        } else {
            res_p = p3;
        }

        if (res_p) {
            sarr_push(&local, res_p);
            if (!is_sentence_boundary_sil) {
                node_pc[node_idx] += 1;
                current_base_consumed += 1;
                while (node_idx < nfpron - 1 && current_base_consumed >= ebc[node_idx]) {
                    current_base_consumed -= ebc[node_idx];
                    node_idx += 1;
                }
            }
        }

        if (prosody_mark) sarr_push(&local, prosody_mark);
    }

    /* append ph[1:-1] to global phones */
    for (int i = 1; i + 1 < local.n; i++) sarr_push(phones, local.a[i]);

    /* word2ph distribution */
    int *w2p_ph = *phs_out; int w2p_n = *phs_n_out; int w2p_cap = *phs_cap_out;
    for (int i = 0; i < nfpron; i++) {
        const char *surface = fstr[i];
        if (strcmp(fpron[i], "IDLE") == 0) continue;
        int total_ph_count = node_pc[i];
        int num_chars = gsv_utf8_len(surface);
        if (num_chars <= 1) {
            sarr_push(words, surface);
            if (w2p_n == w2p_cap) { w2p_cap = w2p_cap ? w2p_cap * 2 : 8; w2p_ph = (int *)realloc(w2p_ph, (size_t)w2p_cap * sizeof(int)); }
            w2p_ph[w2p_n++] = total_ph_count;
        } else {
            int avg_ph = total_ph_count / num_chars;
            int remainder = total_ph_count % num_chars;
            const char *sp = surface;
            for (int cj = 0; cj < num_chars; cj++) {
                uint32_t cp = gsv_utf8_decode(&sp);
                char cb[4]; int cn = gsv_utf8_encode(cp, cb);
                char one[5]; memcpy(one, cb, (size_t)cn); one[cn] = '\0';
                sarr_push(words, one);
                if (w2p_n == w2p_cap) { w2p_cap = w2p_cap ? w2p_cap * 2 : 8; w2p_ph = (int *)realloc(w2p_ph, (size_t)w2p_cap * sizeof(int)); }
                w2p_ph[w2p_n++] = (cj < remainder) ? (avg_ph + 1) : avg_ph;
            }
        }
    }

    sarr_free(&local);
    free(ebc); free(node_pc);
    free_frontend_out(fstr, nfstr, fpron, nfpron, labels, N);

    *phs_out = w2p_ph; *phs_n_out = w2p_n; *phs_cap_out = w2p_cap;
    return 0;
}

/* ---------- public API ---------- */

gsv_g2p_ja *gsv_g2p_ja_create(const char *dicdir) {
    gsv_g2p_ja *j = (gsv_g2p_ja *)calloc(1, sizeof(*j));
    if (!j) return NULL;
    if (!Mecab_initialize(&j->mecab)) { free(j); return NULL; }
    NJD_initialize(&j->njd);
    JPCommon_initialize(&j->jpcommon);
    if (Mecab_load(&j->mecab, dicdir) != TRUE) {
        Mecab_clear(&j->mecab); NJD_clear(&j->njd); JPCommon_clear(&j->jpcommon);
        free(j);
        return NULL;
    }
    j->loaded = 1;
    return j;
}

void gsv_g2p_ja_destroy(gsv_g2p_ja *j) {
    if (!j) return;
    if (j->loaded) {
        res_clear(j);
        free(j->res_strs);
        Mecab_clear(&j->mecab);
        NJD_clear(&j->njd);
        JPCommon_clear(&j->jpcommon);
    }
    free(j);
}

int gsv_g2p_ja_g2p(gsv_g2p_ja *j, const char *text, gsv_g2p_ja_result *out) {
    if (!j || !j->loaded || !text || !out) return -1;
    res_clear(j);

    /* symbols_to_japanese + lower into a fresh buffer */
    char norm[8192];
    int nl = 0;
    const char *p = text;
    while (*p && nl < (int)sizeof(norm) - 4) {
        uint32_t cp = gsv_utf8_decode(&p);
        if (cp == 0xFF05) { /* ％ -> パーセント */
            const char *rep = "パーセント";
            for (const char *r = rep; *r && nl < (int)sizeof(norm) - 1; r++) norm[nl++] = *r;
            continue;
        }
        if (cp >= 'A' && cp <= 'Z') cp += 32;
        char b[4]; int bn = gsv_utf8_encode(cp, b);
        for (int i = 0; i < bn && nl < (int)sizeof(norm) - 1; i++) norm[nl++] = b[i];
    }
    norm[nl] = '\0';

    /* split into JP-run segments + single-char marks.
     * text = seg[0] mark[0] seg[1] mark[1] ... seg[n_segs-1], n_segs = n_marks+1 */
    strarr segs; sarr_init(&segs);
    strarr marks; sarr_init(&marks);
    char run[4096]; int rl = 0;
    run[0] = '\0'; /* invariant: run[rl] is always NUL-terminated */
    p = norm;
    while (*p) {
        uint32_t cp = gsv_utf8_decode(&p);
        if (is_jp_char(cp)) {
            char b[4]; int bn = gsv_utf8_encode(cp, b);
            for (int i = 0; i < bn && rl < (int)sizeof(run) - 1; i++) run[rl++] = b[i];
            run[rl] = '\0';
        } else {
            sarr_push(&segs, run); run[0] = '\0'; rl = 0;
            char mb[4]; int mn = gsv_utf8_encode(cp, mb);
            char mbuf[5]; memcpy(mbuf, mb, (size_t)mn); mbuf[mn] = '\0';
            sarr_push(&marks, mbuf);
        }
    }
    sarr_push(&segs, run);

    strarr phones; sarr_init(&phones);
    strarr words; sarr_init(&words);
    int *phs = NULL; int phs_n = 0, phs_cap = 0;

    for (int i = 0; i < segs.n; i++) {
        if (segs.a[i][0] != '\0') { /* re.match(_japanese_characters) == non-empty JP run */
            g2p_prosody(j, segs.a[i], &phones, &words, &phs, &phs_n, &phs_cap);
        }
        if (i < marks.n) {
            if (strcmp(marks.a[i], " ") == 0) continue; /* skip spaces */
            sarr_push(&phones, marks.a[i]);
            sarr_push(&words, marks.a[i]);
            if (phs_n == phs_cap) { phs_cap = phs_cap ? phs_cap * 2 : 8; phs = (int *)realloc(phs, (size_t)phs_cap * sizeof(int)); }
            phs[phs_n++] = 1;
        }
    }

    /* publish into out (strings already owned by handle via sarr_push->xstrdup? No:
     * phones/words are xstrdup'd in sarr. We hand the arrays to the caller but they
     * must be freed by us on next call. Re-point them into res_strs ownership.) */
    out->phones = (char **)malloc(((size_t)(phones.n > 0 ? phones.n : 1)) * sizeof(char *));
    for (int i = 0; i < phones.n; i++) out->phones[i] = res_intern(j, phones.a[i]);
    out->n_phones = phones.n;

    out->words = (char **)malloc(((size_t)(words.n > 0 ? words.n : 1)) * sizeof(char *));
    for (int i = 0; i < words.n; i++) out->words[i] = res_intern(j, words.a[i]);
    out->n_words = words.n;

    out->phs = (int *)malloc(((size_t)(phs_n > 0 ? phs_n : 1)) * sizeof(int));
    memcpy(out->phs, phs, (size_t)phs_n * sizeof(int));
    out->n_phs = phs_n;

    sarr_free(&segs); sarr_free(&marks); sarr_free(&phones); sarr_free(&words);
    free(phs);
    return 0;
}
