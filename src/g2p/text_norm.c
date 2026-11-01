/* text_norm.c - C port of GPT_SoVITS/G2P/Chinese/Normalization (TextNormalizer).
 * Produces exactly "".join(TextNormalizer().normalize(text)).
 * All fixed string tables come from the generated text_norm_data.h. */
#include "text_norm.h"
#include "text_norm_data.h"
#include <stdlib.h>
#include <string.h>

/* ---------------- working UTF-8 byte buffer ---------------- */
typedef struct { char *a; size_t n, cap; } sbuf;
static void sb_init(sbuf *b) { b->a = (char *)malloc(1); b->a[0] = 0; b->n = 0; b->cap = 1; }
static int sb_need(sbuf *b, size_t extra) {
    if (b->n + extra + 1 <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 8;
    while (nc < b->n + extra + 1) nc *= 2;
    char *nd = (char *)realloc(b->a, nc); if (!nd) return -1;
    b->a = nd; b->cap = nc; return 0;
}
static void sb_append(sbuf *b, const char *s, size_t n) {
    if (sb_need(b, n)) return;
    memcpy(b->a + b->n, s, n); b->n += n; b->a[b->n] = 0;
}
static void sb_append_str(sbuf *b, const char *s) { sb_append(b, s, strlen(s)); }
static void sb_append_cp(sbuf *b, uint32_t cp) { char e[8]; int k = gsv_utf8_encode(cp, e); sb_append(b, e, (size_t)k); }
static void sb_free(sbuf *b) { free(b->a); b->a = NULL; b->n = b->cap = 0; }

/* ---------------- small helpers ---------------- */
static int is_digit_b(char c) { return c >= '0' && c <= '9'; }
static int in_set(uint32_t cp, const uint32_t *set, int n) { for (int i = 0; i < n; i++) if (set[i] == cp) return 1; return 0; }

/* decode codepoint at byte offset i of s[0..n); set *adv to its byte width. */
static uint32_t cp_at(const char *s, size_t n, size_t i, int *adv) {
    if (i >= n) { if (adv) *adv = 0; return 0; }
    const char *p = s + i; uint32_t cp = gsv_utf8_decode(&p);
    if (adv) *adv = (int)(p - (s + i));
    return cp;
}
/* codepoint whose bytes END at offset i (the one immediately before i). 0 if none. */
static uint32_t prev_cp(const char *s, size_t n, size_t i) {
    (void)n;
    if (i == 0) return 0;
    size_t k = i;
    while (k > 0 && ((unsigned char)s[k - 1] & 0xC0) == 0x80) k--;
    const char *p = s + k; uint32_t cp = gsv_utf8_decode(&p); return cp;
}
/* match a single-codepoint literal `lit` at byte offset i of s */
static int lit_at(const char *s, size_t n, size_t i, const char *lit) {
    if (i >= n) return 0;
    int aw, bw; uint32_t ca = cp_at(s, n, i, &aw), cb = cp_at(lit, strlen(lit), 0, &bw);
    return (ca == cb && aw == bw);
}
/* copy exactly one codepoint from s at *i into o, advance *i */
static void copy_cp(sbuf *o, const char *s, size_t n, size_t *i) {
    int adv; (void)cp_at(s, n, *i, &adv);
    sb_append(o, s + *i, (size_t)adv);
    *i += (size_t)adv;
}

/* match (-?(\d+(\.\d+)?))|(\.\d+) at s[i]; returns matched byte length (0 = none).
 * A dangling dot after digits is NOT consumed (regex (\.\d+)? needs a digit). */
static int m_number(const char *s, size_t n, size_t i) {
    size_t j = i;
    if (j < n && s[j] == '-') j++;
    if (j < n && s[j] == '.') {
        size_t k = j + 1;
        if (k >= n || !is_digit_b(s[k])) return 0;
        while (k < n && is_digit_b(s[k])) k++;
        return (int)(k - i);
    }
    if (j >= n || !is_digit_b(s[j])) return 0;
    while (j < n && is_digit_b(s[j])) j++;
    if (j < n && s[j] == '.') {
        size_t k = j + 1;
        if (k < n && is_digit_b(s[k])) {
            while (k < n && is_digit_b(s[k])) k++;
            j = k;
        }
    }
    return (int)(j - i);
}
/* match (-?(\d+)(\.\d+)?) requiring a leading digit (no pure-decimal). */
static int m_signed(const char *s, size_t n, size_t i) {
    size_t j = i;
    if (j < n && s[j] == '-') j++;
    if (j >= n || !is_digit_b(s[j])) return 0;
    while (j < n && is_digit_b(s[j])) j++;
    if (j < n && s[j] == '.') {
        size_t k = j + 1;
        if (k < n && is_digit_b(s[k])) {
            while (k < n && is_digit_b(s[k])) k++;
            j = k;
        }
    }
    return (int)(j - i);
}

/* ---------------- number verbalization (num.py) ---------------- */
static void emit_digit_char(sbuf *o, char d) { sb_append(o, TN_DIGITS + 3 * (d - '0'), 3); }

static void gv(sbuf *o, const char *vs) {
    const char *st = vs; while (*st == '0') st++;
    size_t slen = strlen(st);
    if (slen == 0) return;
    else if (slen == 1) {
        if (slen < strlen(vs)) sb_append(o, TN_DIGITS, 3);   /* 零 */
        emit_digit_char(o, *st);
    } else {
        static const int pw[] = { 8, 4, 3, 2, 1 };
        const char *un[] = { TN_U8, TN_U4, TN_U3, TN_U2, TN_U1 };
        int li = -1;
        for (int p = 0; p < 5; p++) if (pw[p] < (int)slen) { li = p; break; }
        size_t flen = strlen(vs) - (size_t)pw[li];
        char fp[80]; memcpy(fp, vs, flen); fp[flen] = 0;
        gv(o, fp);
        sb_append_str(o, un[li]);
        gv(o, vs + flen);
    }
}
static void vcard(sbuf *o, const char *vs) {
    if (!*vs) return;
    const char *st = vs; while (*st == '0') st++;
    if (!*st) { sb_append(o, TN_DIGITS, 3); return; }   /* 零 */
    sbuf tmp; sb_init(&tmp);
    gv(&tmp, st);
    if (tmp.n >= 6 && memcmp(tmp.a, TN_DIGITS + 3, 3) == 0 && memcmp(tmp.a + 3, TN_U1, 3) == 0)
        sb_append(o, tmp.a + 3, tmp.n - 3);
    else sb_append_str(o, tmp.a);
    sb_free(&tmp);
}
static void vdigit(sbuf *o, const char *vs, int alt_one) {
    for (const char *p = vs; *p; p++)
        if (*p >= '0' && *p <= '9') {
            if (alt_one && *p == '1') sb_append_str(o, TN_YAO);
            else emit_digit_char(o, *p);
        }
}
static void num2str(sbuf *o, const char *vs) {
    const char *dot = strchr(vs, '.');
    char ip[80], dp[80];
    if (dot) { size_t il = (size_t)(dot - vs); memcpy(ip, vs, il); ip[il] = 0; strcpy(dp, dot + 1); }
    else { strcpy(ip, vs); dp[0] = 0; }
    sbuf r; sb_init(&r);
    vcard(&r, ip);
    char dec2[80]; int hasdec = dp[0] != 0;
    if (hasdec) {
        size_t dl = strlen(dp);
        int ended_zero = (dp[dl - 1] == '0');
        while (dl > 0 && dp[dl - 1] == '0') dl--;
        memcpy(dec2, dp, dl);
        if (ended_zero) { dec2[dl] = '0'; dl++; }
        dec2[dl] = 0;
    }
    if (hasdec && dec2[0]) {
        if (r.n == 0) sb_append(&r, TN_DIGITS, 3);   /* 零 */
        sb_append_str(&r, TN_DIAN);
        sbuf dd; sb_init(&dd); vdigit(&dd, dec2, 0); sb_append_str(&r, dd.a); sb_free(&dd);
    }
    sb_append_str(o, r.a);
    sb_free(&r);
}
static void phone2str(sbuf *o, const char *phone, size_t len, int mobile) {
    char buf[512];
    if (len >= sizeof buf) len = sizeof buf - 1;
    memcpy(buf, phone, len); buf[len] = 0;
    if (mobile && buf[0] == '+') memmove(buf, buf + 1, strlen(buf));
    int first = 1;
    char *tok = strtok(buf, mobile ? " " : "-");
    while (tok) {
        if (!first) sb_append_str(o, TN_COMMA);
        sbuf p; sb_init(&p); vdigit(&p, tok, 1); sb_append_str(o, p.a); sb_free(&p);
        first = 0;
        tok = strtok(NULL, mobile ? " " : "-");
    }
}

/* ---------------- time helpers ---------------- */
static int hour_ok(const char *s, size_t n, size_t i, int *hlen) {
    if (i + 1 < n && s[i] == '2' && s[i + 1] >= '0' && s[i + 1] <= '3') { *hlen = 2; return 1; }
    if (i + 1 < n && s[i] >= '0' && s[i] <= '1' && is_digit_b(s[i + 1])) { *hlen = 2; return 1; }
    if (is_digit_b(s[i])) { *hlen = 1; return 1; }
    return 0;
}
static int minsec_ok(const char *s, size_t n, size_t i) {
    return i + 1 < n && s[i] >= '0' && s[i] <= '5' && is_digit_b(s[i + 1]);
}
static void time_num2str(sbuf *o, const char *num, int len) {
    char tmp[8]; memcpy(tmp, num, (size_t)len); tmp[len] = 0;
    const char *st = tmp; while (*st == '0') st++;
    sbuf t; sb_init(&t); num2str(&t, st);
    if (tmp[0] == '0') sb_append(o, TN_DIGITS, 3);   /* 零 */
    sb_append_str(o, t.a); sb_free(&t);
}
static int digits_nonzero(const char *d, int len) { for (int k = 0; k < len; k++) if (d[k] != '0') return 1; return 0; }
/* emit one time at s[i] (known hlen); returns end offset after optional seconds */
static size_t emit_one_time(sbuf *o, const char *s, size_t n, size_t i, int hlen) {
    char hs[3]; memcpy(hs, s + i, (size_t)hlen); hs[hlen] = 0;
    sbuf t; sb_init(&t); num2str(&t, hs); sb_append_str(o, t.a); sb_free(&t); sb_append_str(o, TN_DIAN);
    size_t mj = i + (size_t)hlen + 1;
    if (digits_nonzero(s + mj, 2)) {
        int mv = (s[mj] - '0') * 10 + (s[mj + 1] - '0');
        char mm[3]; memcpy(mm, s + mj, 2);
        if (mv == 30) sb_append_str(o, TN_BAN);
        else { sbuf m; sb_init(&m); time_num2str(&m, mm, 2); sb_append_str(o, m.a); sb_free(&m); sb_append_str(o, TN_FEN); }
    }
    size_t end = mj + 2;
    if (end < n && s[end] == ':' && minsec_ok(s, n, end + 1)) {
        if (digits_nonzero(s + end + 1, 2)) { char ss[3]; memcpy(ss, s + end + 1, 2); sbuf sc; sb_init(&sc); time_num2str(&sc, ss, 2); sb_append_str(o, sc.a); sb_free(&sc); sb_append_str(o, TN_MIAO); }
        end += 3;
    }
    return end;
}

/* ---------------- rules (each: scan in -> out) ---------------- */
static void rule_f2h(sbuf *in, sbuf *out) {
    out->n = 0; const char *p = in->a; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        if (cp >= 0xFF10 && cp <= 0xFF19) cp -= 65248;
        else if ((cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A)) cp -= 65248;
        else if (cp == 0x3000) cp = ' ';
        char e[8]; int k = gsv_utf8_encode(cp, e); sb_append(out, e, (size_t)k);
    }
}

static void rule_date(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    static const char HAO[] = "\xE5\x8F\xB7";   /* 号 U+53F7 */
    while (i < n) {
        size_t j = i, dcnt = 0;
        while (j < n && is_digit_b(s[j])) { j++; dcnt++; }
        int matched = 0; size_t ylen = 0, mstart = 0, mdlen = 0, dstart = 0, ddlen = 0; char hao[4] = ""; size_t total = 0;
        if ((dcnt == 4 || dcnt == 2) && lit_at(s, n, j, TN_NIAN)) {
            ylen = (dcnt == 4) ? 4 : 2;
            size_t k = j + 3;
            mstart = k;
            if (k < n && is_digit_b(s[k])) {
                size_t dj = k; while (dj < n && is_digit_b(s[dj])) dj++;
                size_t dl = dj - k; const char *ds = s + k; int ok = 0;
                if (dl == 1 && ds[0] >= '1' && ds[0] <= '9') ok = 1;
                else if (dl == 2 && ds[0] == '0' && ds[1] >= '1' && ds[1] <= '9') ok = 1;
                else if (dl == 2 && ds[0] == '1' && ds[1] >= '0' && ds[1] <= '2') ok = 1;
                if (ok && dj < n && lit_at(s, n, dj, TN_YUE)) { mdlen = dl; k = dj + 3; }
            }
            dstart = k;
            if (k < n && is_digit_b(s[k])) {
                size_t dj = k; while (dj < n && is_digit_b(s[dj])) dj++;
                size_t dl = dj - k; const char *ds = s + k; int ok = 0;
                if (dl == 1 && ds[0] >= '1' && ds[0] <= '9') ok = 1;
                else if (dl == 2 && ds[0] == '0' && ds[1] >= '1' && ds[1] <= '9') ok = 1;
                else if (dl == 2 && (ds[0] == '1' || ds[0] == '2')) ok = 1;
                else if (dl == 2 && memcmp(ds, "30", 2) == 0) ok = 1;
                else if (dl == 2 && memcmp(ds, "31", 2) == 0) ok = 1;
                if (ok && dj < n && (lit_at(s, n, dj, TN_RI) || memcmp(s + dj, HAO, 3) == 0)) { ddlen = dl; memcpy(hao, s + dj, 3); k = dj + 3; }
            }
            total = k - i; matched = 1;
        }
        if (matched) {
            char ys[8]; memcpy(ys, s + i, ylen); ys[ylen] = 0;
            sbuf t; sb_init(&t); vdigit(&t, ys, 0); sb_append_str(out, t.a); sb_free(&t); sb_append_str(out, TN_NIAN);
            if (mdlen) { char ms[4]; memcpy(ms, s + mstart, mdlen); ms[mdlen] = 0; sbuf t2; sb_init(&t2); vcard(&t2, ms); sb_append_str(out, t2.a); sb_free(&t2); sb_append_str(out, TN_YUE); }
            if (ddlen) { char ds2[4]; memcpy(ds2, s + dstart, ddlen); ds2[ddlen] = 0; sbuf t3; sb_init(&t3); vcard(&t3, ds2); sb_append_str(out, t3.a); sb_free(&t3); sb_append(out, hao, 3); }
            i += total;
        } else copy_cp(out, s, n, &i);
    }
}

static void rule_date2(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int matched = 0; size_t mstart = 0, dstart = 0, total = 0;
        if (i + 4 <= n && is_digit_b(s[i]) && is_digit_b(s[i + 1]) && is_digit_b(s[i + 2]) && is_digit_b(s[i + 3])) {
            char c = s[i + 4];
            if (c == '-' || c == '/' || c == '.' || c == ' ') {
                size_t k = i + 5; int mok = 0;
                if (k + 2 <= n && is_digit_b(s[k]) && is_digit_b(s[k + 1])) {
                    const char *ds = s + k;
                    if ((ds[0] == '0' && ds[1] >= '1' && ds[1] <= '9') || (ds[0] == '1' && ds[1] >= '0' && ds[1] <= '2')) mok = 1;
                }
                if (mok) {
                    size_t d = k + 2;
                    if (d < n && s[d] == c && d + 2 <= n && is_digit_b(s[d + 1]) && is_digit_b(s[d + 2])) {
                        const char *ds = s + d + 1; int dok = 0;
                        if ((ds[0] == '0' && ds[1] >= '1' && ds[1] <= '9') || (ds[0] >= '1' && ds[0] <= '2') || (ds[0] == '3' && (ds[1] == '0' || ds[1] == '1'))) dok = 1;
                        if (dok) { mstart = k; dstart = d + 1; total = (size_t)(d + 3) - i; matched = 1; }
                    }
                }
            }
        }
        if (matched) {
            char ys[5]; memcpy(ys, s + i, 4); ys[4] = 0;
            sbuf t; sb_init(&t); vdigit(&t, ys, 0); sb_append_str(out, t.a); sb_free(&t); sb_append_str(out, TN_NIAN);
            char ms[3]; memcpy(ms, s + mstart, 2); ms[2] = 0; sbuf t2; sb_init(&t2); vcard(&t2, ms); sb_append_str(out, t2.a); sb_free(&t2); sb_append_str(out, TN_YUE);
            char ds2[3]; memcpy(ds2, s + dstart, 2); ds2[2] = 0; sbuf t3; sb_init(&t3); vcard(&t3, ds2); sb_append_str(out, t3.a); sb_free(&t3); sb_append_str(out, TN_RI);
            i += total;
        } else copy_cp(out, s, n, &i);
    }
}

static void rule_time(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int hlen = 0; if (!hour_ok(s, n, i, &hlen)) { copy_cp(out, s, n, &i); continue; }
        size_t mj = i + (size_t)hlen + 1;
        if (s[i + hlen] != ':' || !minsec_ok(s, n, mj)) { copy_cp(out, s, n, &i); continue; }
        size_t end = mj + 2;
        if (end < n && s[end] == ':' && minsec_ok(s, n, end + 1)) end += 3;
        /* range? sep (~ or -) then another time */
        int is_range = 0; size_t t2 = 0; int hlen2 = 0;
        if (end < n && (s[end] == '~' || s[end] == '-') && end + 1 < n) {
            size_t q = end + 1;
            if (hour_ok(s, n, q, &hlen2)) {
                size_t m2 = q + (size_t)hlen2 + 1;
                if (s[q + hlen2] == ':' && minsec_ok(s, n, m2)) { is_range = 1; t2 = q; }
            }
        }
        if (!is_range) { emit_one_time(out, s, n, i, hlen); i = end; continue; }
        /* range */
        size_t e1 = emit_one_time(out, s, n, i, hlen); (void)e1;
        sb_append_str(out, TN_ZHI);
        size_t m2 = t2 + (size_t)hlen2 + 1;
        size_t end2 = m2 + 2;
        if (end2 < n && s[end2] == ':' && minsec_ok(s, n, end2 + 1)) end2 += 3;
        emit_one_time(out, s, n, t2, hlen2);
        i = end2;
    }
}

static void rule_to_range(sbuf *in, sbuf *out) {
    /* RE_TO_RANGE: number unit ~ number unit; first-unit alternatives backtrack
     * (regex semantics): a shorter unit that fails at '~' yields to longer ones. */
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int ml = m_number(s, n, i);
        if (ml > 0) {
            int matched = 0;
            for (int u1 = 0; u1 < N_TO_UNITS && !matched; u1++) {
                size_t L1 = strlen(TO_UNITS[u1]);
                size_t up = i + (size_t)ml;
                if (up + L1 > n || memcmp(s + up, TO_UNITS[u1], L1) != 0) continue;
                size_t tp = up + L1;
                if (tp >= n || s[tp] != '~') continue;
                int ml2 = m_number(s, n, tp + 1);
                if (ml2 <= 0) continue;
                for (int u2 = 0; u2 < N_TO_UNITS; u2++) {
                    size_t L2 = strlen(TO_UNITS[u2]);
                    size_t up2 = tp + 1 + (size_t)ml2;
                    if (up2 + L2 <= n && memcmp(s + up2, TO_UNITS[u2], L2) == 0) {
                        sb_append(out, s + i, tp - i);
                        sb_append_str(out, TN_ZHI);
                        size_t tail = up2 + L2;
                        sb_append(out, s + tp + 1, tail - (tp + 1));
                        i = tail; matched = 1; break;
                    }
                }
            }
            if (matched) continue;
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_temperature(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    const char *UNITS[] = { "\xC2\xB0" "C", "\xE2\x84\x83", TN_DU, TN_SHDU };
    while (i < n) {
        int sign = 0; size_t j = i; if (j < n && s[j] == '-') { sign = 1; j++; }
        int ml = m_number(s, n, j);
        int done = 0;
        if (ml > 0 && is_digit_b(s[j])) {   /* RE_TEMPERATURE: \d+ required after sign */
            for (int u = 0; u < 4; u++) {
                size_t L = strlen(UNITS[u]);
                if (j + (size_t)ml + L <= n && memcmp(s + j + ml, UNITS[u], L) == 0) {
                    char num[80]; memcpy(num, s + j, (size_t)ml); num[ml] = 0;
                    if (sign) sb_append_str(out, TN_LINGXIA);
                    sbuf t; sb_init(&t); num2str(&t, num); sb_append_str(out, t.a); sb_free(&t);
                    /* Python bug: replace_temperature reads group(3)=decimal part as the
                     * unit, so it never equals 摄氏度 and the output is always 度. */
                    sb_append_str(out, TN_DU);
                    i = j + (size_t)ml + L; done = 1; break;
                }
            }
        }
        if (!done) copy_cp(out, s, n, &i);
    }
}

static void rule_measure(sbuf *in, sbuf *out) {
    sbuf a, b; sb_init(&a); sb_append_str(&a, in->a); sb_init(&b);
    for (int k = 0; k < N_MEASURE; k++) {
        const char *key = MEASURE_KEYS[k], *val = MEASURE_VALS[k]; size_t kl = strlen(key);
        b.n = 0; size_t i = 0;
        while (i < a.n) {
            if (i + kl <= a.n && memcmp(a.a + i, key, kl) == 0) { sb_append_str(&b, val); i += kl; }
            else copy_cp(&b, a.a, a.n, &i);
        }
        sbuf t = a; a = b; b = t;
    }
    sb_append_str(out, a.a);
    sb_free(&a); sb_free(&b);
}

static int asmd_operand(const char *s, size_t n, size_t i, int *olen) {
    size_t j = i;
    if (j < n && s[j] == '-') j++;
    int started = 0;
    if (j < n && is_digit_b(s[j])) { while (j < n && is_digit_b(s[j])) j++; started = 1; }
    else if (j < n && s[j] == '.') { j++; if (j >= n || !is_digit_b(s[j])) return 0; while (j < n && is_digit_b(s[j])) j++; started = 1; }
    else if (j < n && ((s[j] >= 'A' && s[j] <= 'Z') || (s[j] >= 'a' && s[j] <= 'z'))) { j++; started = 1; }
    if (!started) return 0;
    while (j < n) { int adv; uint32_t c = cp_at(s, n, j, &adv); if (in_set(c, SUP_KEYS, N_SUP_KEYS)) j += (size_t)adv; else break; }
    *olen = (int)(j - i); return 1;
}
static void asmd_once_pass(sbuf *in, sbuf *out, int *changed) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0; *changed = 0;
    while (i < n) {
        int ol1 = 0;
        if (asmd_operand(s, n, i, &ol1)) {
            size_t op = i + (size_t)ol1; int adv; uint32_t opc = cp_at(s, n, op, &adv);
            const char *opname = NULL;
            if (opc == '+') opname = TN_JIA; else if (opc == '-') opname = TN_JIAN; else if (opc == '=') opname = TN_DENGYU;
            else if (opc == 0x00D7) opname = TN_CHENG; else if (opc == 0x00F7) opname = TN_CHU;
            if (opname && op + (size_t)adv < n) {
                int ol2 = 0;
                if (asmd_operand(s, n, op + (size_t)adv, &ol2)) {
                    sb_append(out, s + i, (size_t)ol1);
                    sb_append_str(out, opname);
                    sb_append(out, s + op + adv, (size_t)ol2);
                    i = op + (size_t)adv + (size_t)ol2; *changed = 1; continue;
                }
            }
        }
        copy_cp(out, s, n, &i);
    }
}
static void rule_asmd(sbuf *in, sbuf *out) {
    sbuf a, b; sb_init(&a); sb_append_str(&a, in->a); sb_init(&b);
    int changed = 1, guard = 0;
    while (changed && guard++ < 64) { changed = 0; asmd_once_pass(&a, &b, &changed); sbuf t = a; a = b; b = t; b.n = 0; }
    sb_append_str(out, a.a);
    sb_free(&a); sb_free(&b);
}

static void rule_power(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int adv; uint32_t c = cp_at(s, n, i, &adv);
        int idx = -1; for (int k = 0; k < N_SUP_KEYS; k++) if (SUP_KEYS[k] == c) { idx = k; break; }
        if (idx >= 0) {
            int run[64]; int rn = 0;
            while (i < n) { int a2; uint32_t cc = cp_at(s, n, i, &a2); int id2 = -1; for (int k = 0; k < N_SUP_KEYS; k++) if (SUP_KEYS[k] == cc) { id2 = k; break; } if (id2 < 0) break; if (rn < 64) run[rn++] = id2; i += (size_t)a2; }
            sb_append_str(out, TN_DE);
            for (int k = 0; k < rn; k++) sb_append(out, SUP_VALS + run[k], 1);
            sb_append_str(out, TN_CIFANG);
            continue;
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_frac(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        /* RE_FRAC: (-?)(\d+)/(\d+) - integer parts only */
        int sign = 0; size_t j = i; if (j < n && s[j] == '-') { sign = 1; j++; }
        int nl = 0;
        while (j + (size_t)nl < n && is_digit_b(s[j + nl])) nl++;
        if (nl > 0 && j + (size_t)nl < n && s[j + nl] == '/') {
            size_t k = j + (size_t)nl + 1; int dl = 0;
            while (k + (size_t)dl < n && is_digit_b(s[k + dl])) dl++;
            if (dl > 0) {
                char num[80], den[80]; memcpy(num, s + j, (size_t)nl); num[nl] = 0; memcpy(den, s + k, (size_t)dl); den[dl] = 0;
                if (sign) sb_append_str(out, TN_FU);
                sbuf d; sb_init(&d); num2str(&d, den); sb_append_str(out, d.a); sb_free(&d); sb_append_str(out, TN_FENZHI);
                sbuf nn; sb_init(&nn); num2str(&nn, num); sb_append_str(out, nn.a); sb_free(&nn);
                i = k + (size_t)dl; continue;
            }
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_percentage(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int sign = 0; size_t j = i; if (j < n && s[j] == '-') { sign = 1; j++; }
        int ml = m_number(s, n, j);
        if (ml > 0 && is_digit_b(s[j]) && j + (size_t)ml < n && s[j + ml] == '%') {
            char num[80]; memcpy(num, s + j, (size_t)ml); num[ml] = 0;
            if (sign) sb_append_str(out, TN_FU);
            sb_append_str(out, TN_BAIFENZHI);
            sbuf t; sb_init(&t); num2str(&t, num); sb_append_str(out, t.a); sb_free(&t);
            i = j + (size_t)ml + 1; continue;
        }
        copy_cp(out, s, n, &i);
    }
}

static int try_mobile_core(const char *s, size_t n, size_t i, size_t *total) {
    if (i >= n || s[i] != '1' || i + 2 > n) return 0;
    char a = s[i + 1], b = s[i + 2]; int ok = 0;
    if (a == '3' || a == '8') ok = 1;
    else if (a == '5' && is_digit_b(b) && b != '4') ok = 1;
    else if (a == '7' && (b == '6' || b == '7' || b == '8')) ok = 1;
    else if (a == '9' && (b == '8' || b == '9')) ok = 1;
    if (!ok) return 0;
    for (int k = 0; k < 8; k++) if (i + 3 + k >= n || !is_digit_b(s[i + 3 + k])) return 0;
    *total = 11; return 1;
}
static void rule_mobile(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        size_t start = i, total = 0;
        size_t j = i;
        if (j < n && s[j] == '+') j++;
        if (j + 2 <= n && s[j] == '8' && s[j + 1] == '6') {
            size_t k = j + 2; if (k < n && s[k] == ' ') k++;
            size_t core = 0;
            if (try_mobile_core(s, n, k, &core)) total = (k - i) + core;
        } else {
            size_t core = 0;
            if (try_mobile_core(s, n, i, &core)) total = core;
        }
        if (total && (i == 0 || !is_digit_b(s[i - 1])) && (i + total >= n || !is_digit_b(s[i + total]))) {
            phone2str(out, s + start, total, 1); i += total; continue;
        }
        copy_cp(out, s, n, &i);
    }
}

static int try_telephone(const char *s, size_t n, size_t i, size_t *total) {
    size_t k = i;
    if (k < n && s[k] == '0') {
        size_t t = k + 1; int arealen = 0;
        if (t + 1 <= n && s[t] == '1' && s[t + 1] == '0') arealen = 3;
        else if (t + 1 <= n && s[t] == '2' && s[t + 1] >= '1' && s[t + 1] <= '3') arealen = 3;
        else if (t + 2 <= n && s[t] >= '3' && s[t] <= '9' && is_digit_b(s[t + 1]) && is_digit_b(s[t + 2])) arealen = 4;
        if (!arealen) return 0;
        k += arealen;
        if (k < n && s[k] == '-') k++;
    }
    if (k >= n || !(s[k] >= '1' && s[k] <= '9')) return 0;
    size_t p = k + 1, cnt = 0;
    while (p < n && is_digit_b(s[p]) && cnt < 7) { p++; cnt++; }
    if (cnt < 6) return 0;
    /* enforce (?!\d): the full local run must be exactly 7 or 8 digits */
    size_t q = k; while (q < n && is_digit_b(s[q])) q++;
    size_t D = q - k;
    if (D != 7 && D != 8) return 0;
    *total = q - i; return 1;
}
static void rule_telephone(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        size_t total = 0;
        if (try_telephone(s, n, i, &total) && (i == 0 || !is_digit_b(s[i - 1])) && (i + total >= n || !is_digit_b(s[i + total]))) {
            phone2str(out, s + i, total, 0); i += total; continue;
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_national(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        if (i + 3 <= n && memcmp(s + i, "400", 3) == 0) {
            size_t k = i + 3; int total = 3;
            if (k < n && s[k] == '-') { k++; total++; }
            if (k + 3 <= n && is_digit_b(s[k]) && is_digit_b(s[k + 1]) && is_digit_b(s[k + 2])) { k += 3; total += 3; } else total = 0;
            if (total) { if (k < n && s[k] == '-') { k++; total++; } if (k + 4 <= n && is_digit_b(s[k]) && is_digit_b(s[k + 1]) && is_digit_b(s[k + 2]) && is_digit_b(s[k + 3])) { k += 4; total += 4; } else total = 0; }
            if (total) { phone2str(out, s + i, (size_t)total, 0); i += total; continue; }
        }
        copy_cp(out, s, n, &i);
    }
}

static void replace_number_emit(sbuf *o, const char *s, size_t i, int ml);
static int in_range_excl(uint32_t c) {
    return is_digit_b((char)c) || c == '+' || c == '-' || c == 0x00D7 || c == 0x00F7 || c == '=';
}
static void rule_range(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int ml1 = m_signed(s, n, i);
        if (ml1 > 0 && i + (size_t)ml1 < n && (s[i + ml1] == '-' || s[i + ml1] == '~')) {
            size_t sep = i + (size_t)ml1;
            int ml2 = m_signed(s, n, sep + 1);
            if (ml2 > 0) {
                size_t end = sep + 1 + (size_t)ml2;
                uint32_t prevc = prev_cp(s, n, i), nextc = cp_at(s, n, end, NULL);
                if (!in_range_excl(prevc) && in_range_excl(nextc)) {
                    /* regex backtracking: give back the second operand's (\.\d+)? */
                    int dotp = -1;
                    for (int t = 0; t < ml2; t++) if (s[sep + 1 + t] == '.') dotp = t;
                    if (dotp > 0) { ml2 = dotp; end = sep + 1 + (size_t)ml2; nextc = cp_at(s, n, end, NULL); }
                }
                if (!in_range_excl(prevc) && !in_range_excl(nextc)) {
                    replace_number_emit(out, s, i, ml1);
                    sb_append_str(out, TN_DAO);
                    replace_number_emit(out, s, sep + 1, ml2);
                    i = end; continue;
                }
            }
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_integer(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        /* RE_INTEGER: (-)(\d+) - digits only, a trailing dot is not consumed */
        if (s[i] == '-') {
            size_t j = i + 1; int nl = 0;
            while (j + (size_t)nl < n && is_digit_b(s[j + nl])) nl++;
            if (nl > 0) { char num[80]; memcpy(num, s + i + 1, (size_t)nl); num[nl] = 0; sb_append_str(out, TN_FU); sbuf t; sb_init(&t); num2str(&t, num); sb_append_str(out, t.a); sb_free(&t); i += 1 + (size_t)nl; continue; }
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_version(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        if (is_digit_b(s[i])) {
            size_t j = i; int dots = 0, ok = 1;
            while (j < n && (is_digit_b(s[j]) || s[j] == '.')) {
                if (s[j] == '.') { dots++; if (j + 1 >= n || !is_digit_b(s[j + 1])) { ok = 0; break; } }
                j++;
            }
            if (ok && dots >= 2) {
                for (size_t t = i; t < j; t++) {
                    if (s[t] == '.') sb_append_str(out, TN_DIAN);
                    else { char c1[2] = { s[t], 0 }; sbuf d; sb_init(&d); num2str(&d, c1); sb_append_str(out, d.a); sb_free(&d); }
                }
                i = j; continue;
            }
        }
        copy_cp(out, s, n, &i);
    }
}

static void replace_number_emit(sbuf *o, const char *s, size_t i, int ml) {
    char num[80]; memcpy(num, s + i, (size_t)ml); num[ml] = 0;
    if (num[0] == '.') { sbuf t; sb_init(&t); num2str(&t, num); sb_append_str(o, t.a); sb_free(&t); return; }
    int sign = (num[0] == '-');
    char body[80]; strcpy(body, sign ? num + 1 : num);
    if (sign) sb_append_str(o, TN_FU);
    sbuf t; sb_init(&t); num2str(&t, body); sb_append_str(o, t.a); sb_free(&t);
}
static void rule_decimal(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int ml = m_number(s, n, i);
        if (ml > 0 && memchr(s + i, '.', (size_t)ml)) { replace_number_emit(out, s, i, ml); i += (size_t)ml; continue; }
        copy_cp(out, s, n, &i);
    }
}
static void rule_number(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        int ml = m_number(s, n, i);
        if (ml > 0) { replace_number_emit(out, s, i, ml); i += (size_t)ml; continue; }
        copy_cp(out, s, n, &i);
    }
}

static void rule_quantifier(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        if (is_digit_b(s[i])) {
            size_t j = i; while (j < n && is_digit_b(s[j])) j++;
            int dlen = (int)(j - i);
            size_t q2 = j; char m2c = 0;
            if (j < n) { int adv; uint32_t c = cp_at(s, n, j, &adv); if (adv > 0 && (c == 0x591A || c == 0x4F59 || c == 0x51E0 || c == '+')) { m2c = (char)c; q2 = j + (size_t)adv; } }
            int qlen = 0;
            for (int u = 0; u < N_QUANTIFIERS; u++) { size_t L = strlen(QUANTIFIERS[u]); if (L && q2 + L <= n && memcmp(s + q2, QUANTIFIERS[u], L) == 0) { qlen = (int)L; break; } }
            if (qlen > 0) {
                char num[80]; memcpy(num, s + i, (size_t)dlen); num[dlen] = 0;
                size_t before = out->n;
                sbuf t; sb_init(&t); num2str(&t, num); sb_append_str(out, t.a); sb_free(&t);
                if (out->n - before == 3 && memcmp(out->a + before, TN_DIGITS + 6, 3) == 0) { out->n = before; sb_append_str(out, TN_LIANG); }   /* 二 -> 两 */
                if (m2c) { if (m2c == '+') sb_append_str(out, TN_DUO); else sb_append(out, s + j, q2 - (size_t)j); }
                sb_append(out, s + q2, (size_t)qlen);
                i = q2 + (size_t)qlen; continue;
            }
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_default(sbuf *in, sbuf *out) {
    const char *s = in->a; size_t n = in->n; out->n = 0; size_t i = 0;
    while (i < n) {
        if (is_digit_b(s[i])) {
            size_t j = i; while (j < n && is_digit_b(s[j])) j++;
            int dlen = (int)(j - i);
            if (dlen >= 3) { char num[128]; memcpy(num, s + i, (size_t)dlen); num[dlen] = 0; sbuf t; sb_init(&t); vdigit(&t, num, 1); sb_append_str(out, t.a); sb_free(&t); i = j; continue; }
        }
        copy_cp(out, s, n, &i);
    }
}

static void rule_post(sbuf *in, sbuf *out) {
    sbuf a, b; sb_init(&a); sb_append_str(&a, in->a); sb_init(&b);
    for (int k = 0; k < N_REPL; k++) {
        const char *key = REPL_KEYS[k], *val = REPL_VALS[k]; size_t kl = strlen(key);
        b.n = 0; size_t i = 0;
        while (i < a.n) {
            if (i + kl <= a.n && memcmp(a.a + i, key, kl) == 0) { sb_append_str(&b, val); i += kl; }
            else copy_cp(&b, a.a, a.n, &i);
        }
        sbuf t = a; a = b; b = t;
    }
    b.n = 0; size_t i = 0;
    while (i < a.n) {
        int adv; uint32_t cp = cp_at(a.a, a.n, i, &adv);
        if (!in_set(cp, POST_REMOVE, N_POST_REMOVE)) sb_append(&b, a.a + i, (size_t)adv);
        i += (size_t)adv;
    }
    sb_append_str(out, b.a);
    sb_free(&a); sb_free(&b);
}

/* ---------------- normalize_sentence: F2H + all rules in Python order ---------------- */
static void normalize_sentence(sbuf *in, sbuf *out) {
    sbuf a, b; sb_init(&a); sb_append_str(&a, in->a); sb_init(&b);
#define STEP(fn) do { fn(&a, &b); sbuf tt = a; a = b; b = tt; b.n = 0; } while (0)
    STEP(rule_f2h);
    STEP(rule_date);
    STEP(rule_date2);
    STEP(rule_time);
    STEP(rule_to_range);
    STEP(rule_temperature);
    STEP(rule_measure);
    STEP(rule_asmd);
    STEP(rule_power);
    STEP(rule_frac);
    STEP(rule_percentage);
    STEP(rule_mobile);
    STEP(rule_telephone);
    STEP(rule_national);
    STEP(rule_range);
    STEP(rule_integer);
    STEP(rule_version);
    STEP(rule_decimal);
    STEP(rule_quantifier);
    STEP(rule_default);
    STEP(rule_number);
    STEP(rule_post);
#undef STEP
    sb_append_str(out, a.a);
    sb_free(&a); sb_free(&b);
}

/* ---------------- _split + orchestration ---------------- */
static int is_ws(uint32_t cp) { return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D || cp == 0x0C || cp == 0x0B || cp == 0x3000; }

static void flush_sentence(sbuf *cur, sbuf *res) {
    size_t s0 = 0, s1 = cur->n;
    while (s0 < s1) { int adv; uint32_t c = cp_at(cur->a, cur->n, s0, &adv); if (!is_ws(c)) break; s0 += (size_t)adv; }
    size_t r = s1;
    while (r > s0) {
        size_t k = r; while (k > s0 && ((unsigned char)cur->a[k - 1] & 0xC0) == 0x80) k--;
        int adv; uint32_t c = cp_at(cur->a, cur->n, k, &adv);
        if (!is_ws(c)) break;
        r = k;
    }
    if (r > s0) {
        sbuf sub; sb_init(&sub); sb_append(&sub, cur->a + s0, r - s0);
        sbuf ns; sb_init(&ns); normalize_sentence(&sub, &ns); sb_append_str(res, ns.a);
        sb_free(&ns); sb_free(&sub);
    }
    cur->n = 0; cur->a[0] = 0;
}

const char *gsv_textnorm_normalize(const char *text, gsv_buf *out) {
    sbuf F; sb_init(&F);
    const char *p = text; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) {
        if (cp == 0x20) continue;
        if (in_set(cp, SPLIT_REMOVE, N_SPLIT_REMOVE)) continue;
        sb_append_cp(&F, cp);
    }
    sbuf cur; sb_init(&cur);
    sbuf res; sb_init(&res);
    size_t i = 0;
    while (i < F.n) {
        int adv; uint32_t c = cp_at(F.a, F.n, i, &adv);
        if (in_set(c, SENT_PUNCT, N_SENT_PUNCT)) {
            sb_append_cp(&cur, c); i += (size_t)adv;
            if (i < F.n) { int qadv; uint32_t q = cp_at(F.a, F.n, i, &qadv); if (in_set(q, SENT_QUOTE, N_SENT_QUOTE)) { sb_append_cp(&cur, q); i += (size_t)qadv; } }
            flush_sentence(&cur, &res);
        } else { sb_append_cp(&cur, c); i += (size_t)adv; }
    }
    flush_sentence(&cur, &res);

    out->len = 0;
    gsv_buf_append(out, res.a, res.n);
    gsv_buf_intern(out, "", 0);
    sb_free(&F); sb_free(&cur); sb_free(&res);
    return out->data;
}
