/* en_textnorm.c - C port of G2P/English/Normalization/expend.py normalize()
 * plus the inflect.number_to_words / ordinal subset it relies on.
 *
 * Faithful to Python semantics:
 *  - re.sub leftmost, non-overlapping, no rescan of replacements within a pass
 *  - RE_INTEGER consumes the preceding \s+ run
 *  - money steps are four sequential passes (pounds_start, pounds_end, dollars_start, dollars_end)
 *  - inflect group=0: zero 3-digit groups contribute nothing but still advance the magnitude index
 *  - inflect mill[0] is a space; double spaces collapse via WHITESPACES.sub(" ")
 */
#include "en_textnorm.h"
#include "en_nfd_data.h"
#include <stdio.h>
#include <ctype.h>

/* ================= unicode helpers ================= */

static int is_ascii_ws(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
static int is_ws_cp(uint32_t cp) {
    if (is_ascii_ws(cp)) return 1;
    if (cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) return 1;
    return 0;
}
/* Python re \w approximation: ascii word chars + common unicode letter/digit ranges. */
static int is_word_cp(uint32_t cp) {
    unsigned char c = (unsigned char)cp;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') return 1;
    if (cp == 0xB2 || cp == 0xB3 || cp == 0xB9) return 1;              /* superscripts */
    if (cp >= 0xBC && cp <= 0xBE) return 1;                            /* vulgar fractions */
    if ((cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6)) return 1;   /* latin-1 letters */
    if (cp >= 0xF8 && cp <= 0x2FF) return 1;                           /* latin ext */
    if (cp >= 0x370 && cp <= 0x3FF) return 1;                          /* greek */
    if (cp >= 0x400 && cp <= 0x4FF) return 1;                          /* cyrillic */
    if (cp >= 0x590 && cp <= 0x5FF) return 1;                          /* hebrew */
    if (cp >= 0x600 && cp <= 0x6FF) return 1;                          /* arabic */
    if (cp >= 0x900 && cp <= 0x97F) return 1;                          /* devanagari */
    if (cp >= 0xE00 && cp <= 0xE7F) return 1;                          /* thai */
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1;                        /* cjk ext a */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;                        /* cjk */
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;                        /* cjk compat */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;                        /* hangul syllables */
    if (cp >= 0x1100 && cp <= 0x11FF) return 1;                        /* hangul jamo */
    if (cp >= 0xFF00 && cp <= 0xFFEF) return 1;                        /* fullwidth forms */
    return 0;
}

/* Codepoint-level boundary checks (byte casts misread UTF-8 lead/continuation bytes). */
static int cp_word_at(const char *s, size_t n, size_t i) {
    if (i >= n) return 0;
    const char *p = s + i;
    uint32_t cp = gsv_utf8_decode(&p);
    return cp != 0 && is_word_cp(cp);
}
static int prev_word_at(const char *s, size_t i) {
    if (i == 0) return 0;
    const char *p = s + i - 1;
    while (p > s && ((unsigned char)*p & 0xC0) == 0x80) p--;
    const char *q = p;
    uint32_t cp = gsv_utf8_decode(&q);
    return cp != 0 && is_word_cp(cp);
}
static int prev_ws_at(const char *s, size_t i) {
    if (i == 0) return 1;
    const char *p = s + i - 1;
    while (p > s && ((unsigned char)*p & 0xC0) == 0x80) p--;
    const char *q = p;
    uint32_t cp = gsv_utf8_decode(&q);
    return cp == 0 || is_ws_cp(cp);
}

static int is_digit_c(char c) { return c >= '0' && c <= '9'; }
static int is_alpha_c(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/* ensure data[len] == 0 without changing len */
static void buf_nul(gsv_buf *b) { if (gsv_buf_reserve(b, 1) == 0) b->data[b->len] = 0; }

/* strip trailing whitespace codepoints from a buffer */
static void buf_strip_trailing_ws(gsv_buf *b) {
    while (b->len > 0) {
        size_t p = b->len - 1;
        while (p > 0 && ((unsigned char)b->data[p] & 0xC0) == 0x80) p--;
        const char *pp = b->data + p;
        uint32_t cp = gsv_utf8_decode(&pp);
        if (!is_ws_cp(cp)) break;
        b->len = p;
    }
}

/* ================= NFD accent stripping ================= */

static int nfd_decomp(uint32_t cp, uint32_t *seq) {
    if (cp < 0x80) { seq[0] = cp; return 1; }
    int lo = 0, hi = GSV_NFD_COUNT - 1, found = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (gsv_nfd_cp[mid] == cp) { found = mid; break; }
        if (gsv_nfd_cp[mid] < cp) lo = mid + 1; else hi = mid - 1;
    }
    if (found < 0) { seq[0] = cp; return 1; }
    int n = gsv_nfd_len[found];
    for (int i = 0; i < n; i++) seq[i] = gsv_nfd_seq[found * GSV_NFD_MAX + i];
    return n;
}

static int is_mn_cp(uint32_t cp) {
    if (cp < 0x80) return 0;
    int lo = 0, hi = GSV_MN_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (gsv_mn_set[mid] == cp) return 1;
        if (gsv_mn_set[mid] < cp) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

/* ================= inflect subset ================= */

static const char *UNIT[10] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
static const char *TEEN[10] = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                               "sixteen", "seventeen", "eighteen", "nineteen"};
static const char *TEN[10]  = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
/* mill[i] for i>=1 (mill[0] is a bare space, i.e. nothing) */
static const char *MILLW[12] = {"", " thousand", " million", " billion", " trillion", " quadrillion",
                                " quintillion", " sextillion", " septillion", " octillion", " nonillion",
                                " decillion"};
/* nth[n%10] for the fallback */
static const char *NTH10[10] = {"th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"};

static int is_nth_suff2(const char *s) {
    return (s[0] == 's' && s[1] == 't') || (s[0] == 'n' && s[1] == 'd') ||
           (s[0] == 'r' && s[1] == 'd') || (s[0] == 't' && s[1] == 'h');
}

/* ordinal() on a digit string: f"{num}{post}" preserving the original digits. */
static void inf_ordinal_digits(const char *digits, gsv_buf *out) {
    size_t L = strlen(digits);
    int last1 = (int)(digits[L - 1] - '0');
    int last2 = (L >= 2) ? ((digits[L - 2] - '0') * 10 + last1) : last1;
    const char *post;
    if (last2 >= 11 && last2 <= 13) post = "th";   /* nth[11..13] */
    else if (last2 <= 9) post = NTH10[last2];      /* nth[0..9] */
    else post = NTH10[last1];                      /* KeyError -> nth[n%10] */
    gsv_buf_append(out, digits, L);
    gsv_buf_append_str(out, post);
}

/* _sub_ord: ordinal of a word string. */
static void inf_ordinal_words(const char *w, gsv_buf *out) {
    size_t L = strlen(w);
    static const struct { const char *suf, *rep; } R[] = {
        {"ty", "tieth"}, {"one", "first"}, {"two", "second"}, {"three", "third"},
        {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}
    };
    for (int i = 0; i < 8; i++) {
        size_t sl = strlen(R[i].suf);
        if (L >= sl && strcmp(w + L - sl, R[i].suf) == 0) {
            gsv_buf_append(out, w, L - sl);
            gsv_buf_append_str(out, R[i].rep);
            return;
        }
    }
    gsv_buf_append_str(out, w);
    gsv_buf_append_str(out, "th");
}

/* words for one non-zero 3-digit group value (hundred/tens/units part). */
static void group_words(int v, int andword, char *s) {
    int h = v / 100, t = (v / 10) % 10, u = v % 10;
    int o = 0;
    if (h) {
        o += (int)sprintf(s + o, "%s hundred", UNIT[h]);
        if (t || u) o += (int)sprintf(s + o, andword ? " and " : "  ");
    }
    if (t == 1) o += (int)sprintf(s + o, "%s", TEEN[u]);
    else if (t > 1) o += (int)sprintf(s + o, "%s%s%s", TEN[t], u ? "-" : "", u ? UNIT[u] : "");
    else if (u) o += (int)sprintf(s + o, "%s", UNIT[u]);
}

/* number_to_words for a digit string (optionally with st/nd/rd/th suffix), group=0.
 * andword: 1 = "and" (inflect default), 0 = "". `digits` need not be NUL-terminated. */
static void inf_num_words(const char *digits, size_t dl, int andword, gsv_buf *out) {
    size_t L = dl;
    if (L == 2 && is_nth_suff2(digits)) { gsv_buf_append_str(out, "zero"); return; }
    int myord = 0;
    const char *d = digits;
    size_t Dl = L;
    if (L >= 2 && is_nth_suff2(d + L - 2)) { myord = 1; Dl -= 2; }
    while (Dl > 1 && d[0] == '0') { d++; Dl--; }
    if (Dl == 1 && d[0] == '0') {
        gsv_buf_append_str(out, myord ? "zeroth" : "zero");
        return;
    }
    int nw = (int)(Dl / 3);      /* count of 3-digit windows */
    int rem = (int)(Dl % 3);     /* leftover (most-significant) length, 0..2 */
    char raw[512];
    size_t rl = 0;
#define APP_CHUNK(words, m) do { \
        const char *_w = (words); \
        rl += (size_t)(int)sprintf(raw + rl, "%s%s, ", _w, \
                                    ((m) >= 1 && (m) <= 11) ? MILLW[m] : " "); \
    } while (0)
    if (rem > 0) {
        int v = 0;
        for (int k = 0; k < rem; k++) v = v * 10 + (d[k] - '0');
        char w[32];
        int t = v / 10, u = v % 10;
        if (t == 1) sprintf(w, "%s", TEEN[u]);
        else if (t > 1) sprintf(w, "%s%s%s", TEN[t], u ? "-" : "", u ? UNIT[u] : "");
        else sprintf(w, "%s", UNIT[u]);
        APP_CHUNK(w, nw);
    }
    for (int g = 0; g < nw; g++) {
        int off = rem + 3 * g;
        int v = (d[off] - '0') * 100 + (d[off + 1] - '0') * 10 + (d[off + 2] - '0');
        if (v == 0) continue;                       /* zero windows emit nothing */
        char w[48];
        group_words(v, andword, w);
        APP_CHUNK(w, nw - 1 - g);                   /* magnitude index counts zero groups */
    }
#undef APP_CHUNK
    /* ---- _handle_chunk post-processing ---- */
    if (rl >= 2 && raw[rl - 2] == ',' && raw[rl - 1] == ' ') rl -= 2;   /* strip trailing ", " */
    char tmp[512];
    size_t tl = 0;
    for (size_t k = 0; k < rl; k++) {              /* WHITESPACES_COMMA: \s+, -> "," */
        if (raw[k] == ',') while (tl > 0 && isspace((unsigned char)tmp[tl - 1])) tl--;
        tmp[tl++] = raw[k];
    }
    /* COMMA_WORD: r", (\S+)\s+\Z" -> f" {andword} \1" (at most one match: the last comma) */
    for (size_t k = 0; k + 2 < tl; k++) {
        if (tmp[k] == ',' && tmp[k + 1] == ' ') {
            size_t p = k + 2, e = p;
            while (e < tl && !isspace((unsigned char)tmp[e])) e++;
            while (e < tl && isspace((unsigned char)tmp[e])) e++;
            if (e == tl && e > p) {
                size_t wl = e - p;
                size_t pre = andword ? 5 : 2;      /* " and " or "  " */
                char rep[128];
                memcpy(rep, andword ? " and " : "  ", pre);
                memcpy(rep + pre, tmp + p, wl);
                memmove(tmp + k, rep, pre + wl);   /* single atomic move (src overlaps dst) */
                tl = k + pre + wl;
                break;
            }
        }
    }
    char fin[512];                                  /* collapse \s+ -> " ", trim */
    size_t fl = 0;
    int inws = 0;
    for (size_t k = 0; k < tl; k++) {
        if (isspace((unsigned char)tmp[k])) { inws = 1; continue; }
        if (inws && fl > 0) fin[fl++] = ' ';
        inws = 0;
        fin[fl++] = tmp[k];
    }
    if (myord) {                                    /* _sub_ord on the last ", "-chunk */
        int last = -1;
        for (size_t k = 0; k + 1 < fl; k++)
            if (fin[k] == ',' && fin[k + 1] == ' ') last = (int)k;
        size_t toff = last >= 0 ? (size_t)last + 2 : 0;
        char tail[512];
        memcpy(tail, fin + toff, fl - toff);
        tail[fl - toff] = 0;
        gsv_buf ob; memset(&ob, 0, sizeof ob);
        inf_ordinal_words(tail, &ob);
        memcpy(fin + toff, ob.data, ob.len);
        fl = toff + ob.len;
        free(ob.data);
    }
    gsv_buf_append(out, fin, fl);
}

/* number_to_words(num, andword="", zero="oh", group=2) as used by _expand_number.
 * Pairs of digits from the left; result joined with a single space
 * (inflect joins with ", " which _expand_number then replaces). */
static void inf_num_words_g2(const char *digits, size_t dl, gsv_buf *out) {
    const char *d = digits;
    size_t off0 = 0;
    while (off0 + 1 < dl && d[0] == '0') { d++; off0++; }
    size_t n = dl - off0;
    int first = 1;
    for (size_t i = 0; i < n; i += 2) {
        if (!first) gsv_buf_append_str(out, " ");
        first = 0;
        if (i + 1 < n) {
            int t = d[i] - '0', u = d[i + 1] - '0';
            if (t == 1) gsv_buf_append_str(out, TEEN[u]);
            else if (t > 1) {
                gsv_buf_append_str(out, TEN[t]);
                if (u) { gsv_buf_append_char(out, '-'); gsv_buf_append_str(out, UNIT[u]); }
            } else if (u) {
                gsv_buf_append_str(out, "oh ");
                gsv_buf_append_str(out, UNIT[u]);
            } else {
                gsv_buf_append_str(out, "oh oh");
            }
        } else {
            int u = d[i] - '0';
            if (u) gsv_buf_append_str(out, UNIT[u]);
            else gsv_buf_append_str(out, "oh");
        }
    }
}

/* ================= pipeline steps ================= */

/* step 1: \b([0-9]+)\.  -> ordinal(digits) + ", " */
static void st_ordinal_number(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (is_digit_c(in[i])) {
            int prev_word = prev_word_at(in, i);
            if (!prev_word) {
                size_t j = i;
                while (j < n && is_digit_c(in[j])) j++;
                if (j + 1 < n && in[j] == '.' && in[j + 1] == ' ') {
                    gsv_buf db; memset(&db, 0, sizeof db);
                    gsv_buf_append(&db, in + i, j - i);
                    buf_nul(&db);
                    inf_ordinal_digits(db.data, out);
                    free(db.data);
                    gsv_buf_append_str(out, ", ");
                    i = j + 2;
                    continue;
                }
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* ---- step 2: RE_ASMD loop ---- */

static int is_sup_cp(uint32_t cp) {
    return cp == 0x2070 || cp == 0xB9 || cp == 0xB2 || cp == 0xB3 ||
           (cp >= 0x2074 && cp <= 0x2079) || cp == 0x2E3 || cp == 0x2B8 || cp == 0x207F;
}
static size_t sup_run_end(const char *s, size_t n, size_t p) {
    size_t r = p;
    while (r < n) {
        const char *pp = s + r;
        uint32_t cp = gsv_utf8_decode(&pp);
        if (!is_sup_cp(cp)) break;
        r = (size_t)(pp - s);
    }
    return r;
}
/* operand matcher: (-?)((\d+)(\.\d+)?[sup]*) | (\.\d+[sup]*) | ([A-Za-z][sup]*). Returns length or 0. */
static int asmd_operand(const char *s, size_t n, size_t p) {
    {
        size_t q = p;
        if (q < n && s[q] == '-') q++;
        if (q < n && is_digit_c(s[q])) {
            size_t r = q;
            while (r < n && is_digit_c(s[r])) r++;
            if (r + 1 < n && s[r] == '.' && is_digit_c(s[r + 1])) {
                r++;
                while (r < n && is_digit_c(s[r])) r++;
            }
            return (int)(sup_run_end(s, n, r) - p);
        }
    }
    if (p + 1 < n && s[p] == '.' && is_digit_c(s[p + 1])) {
        size_t r = p + 2;
        while (r < n && is_digit_c(s[r])) r++;
        return (int)(sup_run_end(s, n, r) - p);
    }
    if (p < n && is_alpha_c(s[p]))
        return (int)(sup_run_end(s, n, p + 1) - p);
    return 0;
}

static const char *asmd_op_map(uint32_t op) {
    switch (op) {
        case '+': return " plus ";
        case '-': return " minus ";
        case 0xD7: return " times ";
        case 0xF7: return " divided by ";
        case '=': return " Equals ";
    }
    return NULL;
}

static void st_asmd(const char *in, gsv_buf *out) {
    gsv_buf cur; memset(&cur, 0, sizeof cur);
    gsv_buf_append_str(&cur, in);
    for (;;) {
        buf_nul(&cur);
        const char *s = cur.data;
        size_t n = cur.len;
        gsv_buf nxt; memset(&nxt, 0, sizeof nxt);
        int found = 0;
        size_t i = 0;
        while (i < n) {
            int lhs = asmd_operand(s, n, i);
            if (lhs > 0) {
                size_t q = i + (size_t)lhs;
                size_t w0 = q;
                while (q < n) { const char *wp = s + q; uint32_t wcp = gsv_utf8_decode(&wp); if (!is_ws_cp(wcp)) break; q = (size_t)(wp - s); }
                if (q > w0 && q < n) {
                    const char *pp = s + q;
                    uint32_t op = gsv_utf8_decode(&pp);
                    const char *m = asmd_op_map(op);
                    size_t oe = (size_t)(pp - s);
                    if (m && oe > q) {
                        size_t w1 = oe;
                        while (w1 < n) { const char *wp = s + w1; uint32_t wcp = gsv_utf8_decode(&wp); if (!is_ws_cp(wcp)) break; w1 = (size_t)(wp - s); }
                        if (w1 > oe) {
                            int rhs = asmd_operand(s, n, w1);
                            if (rhs > 0) {
                                gsv_buf_append(&nxt, s + i, (size_t)lhs);
                                gsv_buf_append_str(&nxt, m);
                                gsv_buf_append(&nxt, s + w1, (size_t)rhs);
                                found = 1;
                                i = w1 + (size_t)rhs;
                                continue;
                            }
                        }
                    }
                }
            }
            gsv_buf_append_char(&nxt, s[i]);
            i++;
        }
        free(cur.data);
        cur = nxt;
        if (!found) break;
    }
    gsv_buf_append(out, cur.data ? cur.data : "", cur.len);
    free(cur.data);
}

/* step 3: (?:^|\s+)(-)(\d+) -> "negative "+digits ; the \s+ run is consumed. */
static void st_negative(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (in[i] == '-' && i + 1 < n && is_digit_c(in[i + 1])) {
            int ok = prev_ws_at(in, i);
            if (ok) {
                buf_strip_trailing_ws(out);
                gsv_buf_append_str(out, "negative ");
                size_t j = i + 1;
                while (j < n && is_digit_c(in[j])) j++;
                gsv_buf_append(out, in + i + 1, j - i - 1);
                i = j;
                continue;
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 4: ([0-9][0-9,]+[0-9]) -> strip commas */
static void st_comma_number(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (is_digit_c(in[i])) {
            size_t j = i;
            while (j < n && (is_digit_c(in[j]) || in[j] == ',')) j++;
            if (j - i >= 3 && is_digit_c(in[j - 1])) {
                for (size_t k = i; k < j; k++)
                    if (in[k] != ',') gsv_buf_append_char(out, in[k]);
                i = j;
                continue;
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 5: \b([01]?[0-9]|2[0-3]):([0-5][0-9])\b */
static int try_time(const char *in, size_t n, size_t i, int *h, int *m) {
    size_t q;
    int hours, min;
    /* alt 1a: [01][0-9] */
    if (i + 4 < n && (in[i] == '0' || in[i] == '1') && is_digit_c(in[i + 1]) && in[i + 2] == ':') {
        hours = (in[i] - '0') * 10 + (in[i + 1] - '0');
        q = i + 3;
        if (q + 1 < n && in[q] >= '0' && in[q] <= '5' && is_digit_c(in[q + 1])) {
            min = (in[q] - '0') * 10 + (in[q + 1] - '0');
            size_t e = q + 2;
            if (!cp_word_at(in, n, e)) { *h = hours; *m = min; return (int)e; }
        }
    }
    /* alt 1b: [0-9] single */
    if (i + 3 < n && in[i + 1] == ':') {
        hours = in[i] - '0';
        q = i + 2;
        if (q + 1 < n && in[q] >= '0' && in[q] <= '5' && is_digit_c(in[q + 1])) {
            min = (in[q] - '0') * 10 + (in[q + 1] - '0');
            size_t e = q + 2;
            if (!cp_word_at(in, n, e)) { *h = hours; *m = min; return (int)e; }
        }
    }
    /* alt 2: 2[0-3] */
    if (i + 4 < n && in[i] == '2' && in[i + 1] >= '0' && in[i + 1] <= '3' && in[i + 2] == ':') {
        hours = 20 + (in[i + 1] - '0');
        q = i + 3;
        if (q + 1 < n && in[q] >= '0' && in[q] <= '5' && is_digit_c(in[q + 1])) {
            min = (in[q] - '0') * 10 + (in[q + 1] - '0');
            size_t e = q + 2;
            if (!cp_word_at(in, n, e)) { *h = hours; *m = min; return (int)e; }
        }
    }
    return -1;
}

static void st_time(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        int boundary = !prev_word_at(in, i);
        if (boundary && is_digit_c(in[i])) {
            int h, m;
            int e = try_time(in, n, i, &h, &m);
            if (e > 0) {
                const char *period = (h < 12) ? "a.m." : "p.m.";
                if (h > 12) h -= 12;
                char nb[16];
                sprintf(nb, "%d", h);
                gsv_buf hb; memset(&hb, 0, sizeof hb);
                inf_num_words(nb, strlen(nb), 1, &hb);
                gsv_buf_append(out, hb.data, hb.len);
                if (m == 0) {
                    gsv_buf_append_str(out, " o'clock ");
                } else {
                    gsv_buf_append_char(out, ' ');
                    char mb[16];
                    sprintf(mb, "%d", m);
                    gsv_buf mb2; memset(&mb2, 0, sizeof mb2);
                    inf_num_words(mb, strlen(mb), 1, &mb2);
                    gsv_buf_append(out, mb2.data, mb2.len);
                    free(mb2.data);
                    gsv_buf_append_char(out, ' ');
                }
                gsv_buf_append_str(out, period);
                free(hb.data);
                i = (size_t)e;
                continue;
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 6: \b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b */
static const struct { const char *unit; const char *sing; const char *plur; } MEAS[12] = {
    {"m", "meter", "meters"},
    {"km", "kilometer", "kilometers"},
    {"km/h", "kilometer per hour", "kilometers per hour"},
    {"ft", "feet", "feet"},
    {"L", "liter", "liters"},
    {"tbsp", "tablespoon", "tablespoons"},
    {"tsp", "teaspoon", "teaspoons"},
    {"h", "hour", "hours"},
    {"min", "minute", "minutes"},
    {"s", "second", "seconds"},
    {"\xC2\xB0" "C", "degree celsius", "degrees celsius"},
    {"\xC2\xB0" "F", "degree fahrenheit", "degrees fahrenheit"},
};

static void st_measurement(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        int boundary = !prev_word_at(in, i);
        if (boundary && is_digit_c(in[i])) {
            size_t q = i;
            while (q < n && is_digit_c(in[q])) q++;
            int has_dec = 0;
            if (q + 1 < n && in[q] == '.' && is_digit_c(in[q + 1])) {
                has_dec = 1;
                q++;
                while (q < n && is_digit_c(in[q])) q++;
            }
            for (int u = 0; u < 12; u++) {
                const char *unit = MEAS[u].unit;
                size_t ul = strlen(unit);
                if (q + ul <= n && memcmp(in + q, unit, ul) == 0) {
                    size_t e = q + ul;
                    if (!cp_word_at(in, n, e)) {
                        long long num = 0;
                        for (size_t k = i; k < q; k++)
                            if (in[k] != '.') num = num * 10 + (in[k] - '0');
                        int ptr = (!has_dec && num == 1) ? 0 : 1;
                        gsv_buf_append(out, in + i, q - i);   /* original number text */
                        gsv_buf_append_char(out, ' ');
                        gsv_buf_append_str(out, ptr ? MEAS[u].plur : MEAS[u].sing);
                        i = e;
                        goto meas_done;
                    }
                }
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
        meas_done:;
    }
}

/* ---- step 7: money (four sequential passes) ---- */

static void expand_money(const char *numstr, int dollars, gsv_buf *out) {
    const char *parts[8];
    size_t plen[8];
    int nparts = 0;
    size_t off = 0, L = strlen(numstr);
    for (size_t k = 0;; k++) {
        if (k == L || numstr[k] == '.') {
            parts[nparts] = numstr + off;
            plen[nparts] = k - off;
            nparts++;
            if (k == L) break;
            off = k + 1;
        }
    }
    const char *u1 = dollars ? "dollar" : "pound";
    const char *u1p = dollars ? "dollars" : "pounds";
    const char *u2 = dollars ? "cent" : "penny";
    const char *u2p = dollars ? "cents" : "pence";
    if (nparts > 2) {
        gsv_buf_append(out, numstr, L);
        gsv_buf_append_str(out, " ");
        gsv_buf_append_str(out, u1p);
        return;
    }
    long long a = 0;
    int a_ok = 1;   /* empty parts[0] means 0, like Python `int(parts[0]) if parts[0] else 0` */
    for (size_t k = 0; k < plen[0]; k++) {
        char c = numstr[k];
        if (!is_digit_c(c)) { a_ok = 0; break; }
        a = a * 10 + (c - '0');
    }
    long long b = 0;
    int b_ok = 1;
    if (nparts > 1 && plen[1] > 0) {
        for (size_t k = 0; k < plen[1]; k++) {
            char c = numstr[off + k];
            if (!is_digit_c(c)) { b_ok = 0; break; }
            b = b * 10 + (c - '0');
        }
        size_t l = plen[1];
        while (l < 2) { b *= 10; l++; }   /* ljust(2, "0") */
    }
    if (!a_ok || !b_ok) {   /* reference would raise; keep raw first part */
        gsv_buf_append(out, parts[0], plen[0]);
        gsv_buf_append_str(out, " ");
        gsv_buf_append_str(out, u1p);
        return;
    }
    if (a && b) {
        char nb[64];
        sprintf(nb, "%lld %s and %lld %s", a, a == 1 ? u1 : u1p, b, b == 1 ? u2 : u2p);
        gsv_buf_append_str(out, nb);
    } else if (a) {
        char nb[32];
        sprintf(nb, "%lld %s", a, a == 1 ? u1 : u1p);
        gsv_buf_append_str(out, nb);
    } else if (b) {
        char nb[32];
        sprintf(nb, "%lld %s", b, b == 1 ? u2 : u2p);
        gsv_buf_append_str(out, nb);
    } else {
        gsv_buf_append_str(out, dollars ? "zero dollars" : "zero pounds");
    }
}

static void money_pass_start(const char *in, size_t n, const char *sym, size_t symlen, int dollars, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (n - i >= symlen && memcmp(in + i, sym, symlen) == 0) {
            size_t q = i + symlen;
            size_t r = q;
            while (r < n && (is_digit_c(in[r]) || in[r] == '.' || in[r] == ',')) r++;
            size_t e = r;
            while (e > q && !is_digit_c(in[e - 1])) e--;   /* group must end on a digit */
            if (e > q) {
                gsv_buf nb; memset(&nb, 0, sizeof nb);
                gsv_buf_append(&nb, in + q, e - q);
                buf_nul(&nb);
                expand_money(nb.data, dollars, out);
                free(nb.data);
                i = e;
                continue;
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

static void money_pass_end(const char *in, size_t n, const char *sym, size_t symlen,
                           int dollars, int dollar_paren, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (n - i >= symlen && memcmp(in + i, sym, symlen) == 0 &&
            i > 0 && is_digit_c(in[i - 1])) {
            size_t j = i;
            while (j > 0) {
                char c = in[j - 1];
                if (is_digit_c(c) || c == '.' || c == ',' || (dollar_paren && c == '(')) j--;
                else break;
            }
            /* in[j..i) was already copied to out byte-by-byte: drop it */
            size_t cut = i - j;
            if (out->len >= cut) out->len -= cut;
            gsv_buf nb; memset(&nb, 0, sizeof nb);
            gsv_buf_append(&nb, in + j, i - j);
            buf_nul(&nb);
            expand_money(nb.data, dollars, out);
            free(nb.data);
            i += symlen;
            continue;
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

static void st_money(const char *in, size_t n, gsv_buf *out) {
    gsv_buf p1, p2, p3, p4;
    memset(&p1, 0, sizeof p1); memset(&p2, 0, sizeof p2);
    memset(&p3, 0, sizeof p3); memset(&p4, 0, sizeof p4);
    money_pass_start(in, n, "\xC2\xA3", 2, 0, &p1);
    buf_nul(&p1);
    money_pass_end(p1.data, p1.len, "\xC2\xA3", 2, 0, 0, &p2);
    free(p1.data);
    buf_nul(&p2);
    money_pass_start(p2.data, p2.len, "$", 1, 1, &p3);
    free(p2.data);
    buf_nul(&p3);
    money_pass_end(p3.data, p3.len, "$", 1, 1, 1, &p4);
    free(p3.data);
    gsv_buf_append(out, p4.data ? p4.data : "", p4.len);
    free(p4.data);
}

/* step 8: ([0-9]+\.\s*[0-9]+) -> parts[0] + " point " + join(chars of rest, " ") */
static void st_decimal(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (is_digit_c(in[i])) {
            size_t q = i;
            while (q < n && is_digit_c(in[q])) q++;
            if (q < n && in[q] == '.') {
                size_t w = q + 1;
                while (w < n) { const char *wp = in + w; uint32_t wcp = gsv_utf8_decode(&wp); if (!is_ws_cp(wcp)) break; w = (size_t)(wp - in); }
                if (w < n && is_digit_c(in[w])) {
                    size_t r = w;
                    while (r < n && is_digit_c(in[r])) r++;
                    gsv_buf_append(out, in + i, q - i);
                    gsv_buf_append_str(out, " point ");
                    for (size_t k = q + 1; k < r; k++) {
                        if (k > q + 1) gsv_buf_append_char(out, ' ');
                        if (in[k] == '.') gsv_buf_append_str(out, "point");
                        else gsv_buf_append_char(out, in[k]);
                    }
                    i = r;
                    continue;
                }
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 9: ([0-9]+/[0-9]+) */
static void st_fraction(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (is_digit_c(in[i])) {
            size_t q = i;
            while (q < n && is_digit_c(in[q])) q++;
            if (q < n && in[q] == '/' && q + 1 < n && is_digit_c(in[q + 1])) {
                size_t r = q + 1;
                while (r < n && is_digit_c(in[r])) r++;
                /* zero-stripped views for comparisons */
                const char *ns = in + i;
                size_t nl = q - i;
                while (nl > 1 && ns[0] == '0') { ns++; nl--; }
                const char *ds = in + q + 1;
                size_t dl = r - q - 1;
                while (dl > 1 && ds[0] == '0') { ds++; dl--; }
                int den_is_1 = (dl == 1 && ds[0] == '1');
                int den_is_2 = (dl == 1 && ds[0] == '2');
                int num_gt_1 = !(nl == 1 && (ns[0] == '0' || ns[0] == '1'));

                gsv_buf npart; memset(&npart, 0, sizeof npart);
                inf_num_words(in + i, q - i, 1, &npart);   /* leading zeros handled inside */
                if (den_is_1) {
                    gsv_buf_append(out, npart.data, npart.len);
                } else if (den_is_2) {
                    gsv_buf_append(out, npart.data, npart.len);
                    gsv_buf_append_str(out, (nl == 1 && ns[0] == '1') ? " half" : " halves");
                } else {
                    gsv_buf dw; memset(&dw, 0, sizeof dw);
                    inf_num_words(in + q + 1, r - q - 1, 1, &dw);
                    buf_nul(&dw);
                    gsv_buf dw2; memset(&dw2, 0, sizeof dw2);
                    inf_ordinal_words(dw.data, &dw2);
                    free(dw.data);
                    gsv_buf_append(out, npart.data, npart.len);
                    gsv_buf_append_char(out, ' ');
                    gsv_buf_append(out, dw2.data, dw2.len);
                    if (num_gt_1) gsv_buf_append_char(out, 's');
                    free(dw2.data);
                }
                free(npart.data);
                i = r;
                continue;
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 10: [0-9]+(st|nd|rd|th) -> number_to_words(match) */
static void st_ordinal(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (is_digit_c(in[i])) {
            size_t q = i;
            while (q < n && is_digit_c(in[q])) q++;
            if (q + 2 <= n && is_nth_suff2(in + q)) {
                gsv_buf mb; memset(&mb, 0, sizeof mb);
                gsv_buf_append(&mb, in + i, q - i + 2);
                buf_nul(&mb);
                inf_num_words(mb.data, mb.len, 1, out);
                free(mb.data);
                i = q + 2;
                continue;
            }
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 11: [0-9]+ -> _expand_number */
static void st_number(const char *in, size_t n, gsv_buf *out) {
    size_t i = 0;
    while (i < n) {
        if (is_digit_c(in[i])) {
            size_t q = i;
            while (q < n && is_digit_c(in[q])) q++;
            const char *d = in + i;
            size_t L = q - i;
            while (L > 1 && d[0] == '0') { d++; L--; }
            int gt1000, lt3000;
            if (L < 4) { gt1000 = 0; lt3000 = 1; }
            else if (L > 4) { gt1000 = 1; lt3000 = 0; }
            else { gt1000 = strncmp(d, "1000", L) > 0; lt3000 = strncmp(d, "3000", L) < 0; }
            if (gt1000 && lt3000) {
                if (strncmp(d, "2000", L) == 0) {
                    gsv_buf_append_str(out, "two thousand");
                } else if (d[0] == '2' && d[1] == '0' && d[2] == '0') {
                    /* 2001..2009 */
                    gsv_buf_append_str(out, "two thousand ");
                    char t[8];
                    sprintf(t, "%c", d[3]);
                    inf_num_words(t, strlen(t), 1, out);
                } else if (d[L - 1] == '0' && d[L - 2] == '0') {
                    gsv_buf hb; memset(&hb, 0, sizeof hb);
                    gsv_buf_append(&hb, d, L - 2);
                    buf_nul(&hb);
                    inf_num_words(hb.data, hb.len, 1, out);
                    free(hb.data);
                    gsv_buf_append_str(out, " hundred");
                } else {
                    inf_num_words_g2(d, L, out);
                }
            } else {
                inf_num_words(d, L, 0, out);
            }
            i = q;
            continue;
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 12: NFD + drop Mn */
static void st_nfd_strip(const char *in, size_t n, gsv_buf *out) {
    const char *p = in;
    while ((size_t)(p - in) < n) {
        uint32_t cp = gsv_utf8_decode(&p);
        if (cp == 0) break;
        uint32_t seq[GSV_NFD_MAX];
        int nseq = nfd_decomp(cp, seq);
        for (int i = 0; i < nseq; i++) {
            if (is_mn_cp(seq[i])) continue;
            char eb[8];
            int e = gsv_utf8_encode(seq[i], eb);
            gsv_buf_append(out, eb, (size_t)e);
        }
    }
}

/* step 13: % -> " percent" */
static void st_percent(const char *in, size_t n, gsv_buf *out) {
    for (size_t i = 0; i < n; i++) {
        if (in[i] == '%') gsv_buf_append_str(out, " percent");
        else gsv_buf_append_char(out, in[i]);
    }
}

/* step 14/15: (?i)i\.e\. -> "that is" ; (?i)e\.g\. -> "for example" */
static void st_ie_eg(const char *in, size_t n, int eg, gsv_buf *out) {
    const char a = (char)(eg ? 'e' : 'i'), b = (char)(eg ? 'g' : 'e');
    size_t i = 0;
    while (i < n) {
        unsigned char c0 = (unsigned char)in[i];
        if (i + 4 <= n &&
            (c0 == a || c0 == (unsigned char)(a - 'a' + 'A')) && in[i + 1] == '.' &&
            ((unsigned char)in[i + 2] == b || (unsigned char)in[i + 2] == (unsigned char)(b - 'a' + 'A')) &&
            in[i + 3] == '.') {
            gsv_buf_append_str(out, eg ? "for example" : "that is");
            i += 4;
            continue;
        }
        gsv_buf_append_char(out, in[i]);
        i++;
    }
}

/* step 16: (?<!^)(?<![\s])([A-Z]) -> " \1" */
static void st_upper_split(const char *in, size_t n, gsv_buf *out) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c >= 'A' && c <= 'Z' && !prev_ws_at(in, i))
            gsv_buf_append_char(out, ' ');
        gsv_buf_append_char(out, in[i]);
    }
}

/* ================= entry point ================= */

int gsv_en_textnorm_normalize(const char *text, gsv_buf *out) {
    out->len = 0;
    gsv_buf b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16;
    memset(&b1, 0, sizeof b1); memset(&b2, 0, sizeof b2); memset(&b3, 0, sizeof b3);
    memset(&b4, 0, sizeof b4); memset(&b5, 0, sizeof b5); memset(&b6, 0, sizeof b6);
    memset(&b7, 0, sizeof b7); memset(&b8, 0, sizeof b8); memset(&b9, 0, sizeof b9);
    memset(&b10, 0, sizeof b10); memset(&b11, 0, sizeof b11); memset(&b12, 0, sizeof b12);
    memset(&b13, 0, sizeof b13); memset(&b14, 0, sizeof b14); memset(&b15, 0, sizeof b15);
    memset(&b16, 0, sizeof b16);

    size_t n = strlen(text);
    st_ordinal_number(text, n, &b1);          buf_nul(&b1);
    st_asmd(b1.data, &b2);                    free(b1.data);  buf_nul(&b2);
    st_negative(b2.data, b2.len, &b3);        free(b2.data);  buf_nul(&b3);
    st_comma_number(b3.data, b3.len, &b4);    free(b3.data);  buf_nul(&b4);
    st_time(b4.data, b4.len, &b5);            free(b4.data);  buf_nul(&b5);
    st_measurement(b5.data, b5.len, &b6);     free(b5.data);  buf_nul(&b6);
    st_money(b6.data, b6.len, &b7);           free(b6.data);  buf_nul(&b7);
    st_decimal(b7.data, b7.len, &b8);         free(b7.data);  buf_nul(&b8);
    st_fraction(b8.data, b8.len, &b9);        free(b8.data);  buf_nul(&b9);
    st_ordinal(b9.data, b9.len, &b10);        free(b9.data);  buf_nul(&b10);
    st_number(b10.data, b10.len, &b11);       free(b10.data); buf_nul(&b11);
    st_nfd_strip(b11.data, b11.len, &b12); free(b11.data); buf_nul(&b12);
    st_percent(b12.data, b12.len, &b13);   free(b12.data);  buf_nul(&b13);
    st_ie_eg(b13.data, b13.len, 0, &b14);  free(b13.data);  buf_nul(&b14);
    st_ie_eg(b14.data, b14.len, 1, &b15);  free(b14.data);  buf_nul(&b15);
    st_upper_split(b15.data, b15.len, &b16); free(b15.data);

    gsv_buf_append(out, b16.data ? b16.data : "", b16.len);
    if (gsv_buf_reserve(out, 1) == 0) out->data[out->len] = 0;
    free(b16.data);
    return 0;
}
