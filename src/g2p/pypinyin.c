/* pypinyin.c - C port of pypinyin.lazy_pinyin (INITIALS + FINALS_TONE3, neutral_tone_with_five). */
#include "pypinyin.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ---------------- small codepoint-string type ---------------- */
typedef struct { uint32_t cp[24]; int n; } pys;

static void pys_clear(pys *p) { p->n = 0; }
static void pys_add(pys *p, uint32_t c) { if (p->n < 24) p->cp[p->n++] = c; }
static void pys_from_utf8(pys *p, const char *s) {
    pys_clear(p); const char *q = s; uint32_t c;
    while ((c = gsv_utf8_decode(&q)) != 0) pys_add(p, c);
}
static int pys_to_utf8(const pys *p, char *out) {
    int o = 0; for (int i = 0; i < p->n; i++) o += gsv_utf8_encode(p->cp[i], out + o);
    out[o] = 0; return o;
}
static int pys_startswith_ascii(const pys *p, const char *a) {
    int n = (int)strlen(a);
    if (p->n < n) return 0;
    for (int i = 0; i < n; i++) if (p->cp[i] != (unsigned char)a[i]) return 0;
    return 1;
}

/* ---------------- tone-mark tables ---------------- */
static int is_u_tone(uint32_t c) {
    return c=='u' || c==0x016B || c==0x00FA || c==0x01D4 || c==0x00F9;
}
static int is_i_tone(uint32_t c) {
    return c=='i' || c==0x012B || c==0x00ED || c==0x01D0 || c==0x00EC;
}
static uint32_t uv_map(uint32_t c) {
    switch (c) { case 'u': return 0x00FC; case 0x016B: return 0x1D96;
                 case 0x00FA: return 0x1D98; case 0x01D4: return 0x1D9A;
                 case 0x00F9: return 0x1D9C; }
    return c;
}
/* toned char -> tone digit (1-4) or 0 */
static int tone_digit(uint32_t c) {
    switch (c) {
        case 0x0101: case 0x0113: case 0x014D: case 0x012B: case 0x016B:
        case 0x0196: return 1;   /* a e o i u tones + ǖ */
        case 0x00E1: case 0x00E9: case 0x00F3: case 0x00ED: case 0x00FA:
        case 0x0144: case 0x1E3F: case 0x1EB9: case 0x01D8: return 2; /* + ǘ */
        case 0x01CE: case 0x011B: case 0x01D2: case 0x01D0: case 0x01D4:
        case 0x0148: case 0x01DA: return 3;   /* + ǚ */
        case 0x00E0: case 0x00E8: case 0x00F2: case 0x00EC: case 0x00F9:
        case 0x0199: case 0x01DC: return 4;   /* + ǜ */
    }
    return 0;
}
/* toned char -> unmarked base (keeps ü as ü) */
static uint32_t unmark(uint32_t c) {
    int d = tone_digit(c);
    (void)d;
    switch (c) {
        case 0x0101: case 0x00E1: case 0x01CE: case 0x00E0: return 'a';
        case 0x0113: case 0x00E9: case 0x011B: case 0x00E8: return 'e';
        case 0x014D: case 0x00F3: case 0x01D2: case 0x00F2: return 'o';
        case 0x012B: case 0x00ED: case 0x01D0: case 0x00EC: return 'i';
        case 0x016B: case 0x00FA: case 0x01D4: case 0x00F9: return 'u';
        case 0x00FC: case 0x0196: case 0x01D8: case 0x01DA: case 0x01DC:
        case 0x1D96: case 0x1D98: case 0x1D9A: case 0x1D9C: return 0x00FC; /* ü */
        case 0x0144: case 0x0148: case 0x0199: return 'n';
        case 0x1E3F: return 'm';
        case 0x1EB9: return 0x00EA; /* ê */
    }
    return c;
}

/* ---------------- _FINALS membership (no-tone, ü preserved) ---------------- */
static const char *FINALS[] = {
    "i","u","\xC3\xBC","a","ia","ua","o","uo","e","ie","\xC3\xBC""e","ai","uai","ei",
    "uei","ao","iao","ou","iou","an","ian","uan","\xC3\xBC""an","en","in","uen","\xC3\xBC""n",
    "ang","iang","uang","eng","ing","ueng","ong","iong","er","\xC3\xAA"
};
#define NFINALS (int)(sizeof(FINALS)/sizeof(FINALS[0]))
static int in_finals(const pys *f) {
    char buf[16]; int n = pys_to_utf8(f, buf);
    for (int i = 0; i < NFINALS; i++) if ((int)strlen(FINALS[i]) == n && strcmp(buf, FINALS[i]) == 0) return 1;
    return 0;
}

/* ---------------- convert_finals sub-steps (operate on pys) ---------------- */
static void cf_zero_consonant(pys *p) {
    pys raw = *p;
    if (p->n > 0 && p->cp[0] == 'y') {
        uint32_t first = (p->n > 1) ? p->cp[1] : 0;
        pys r; pys_clear(&r);
        if (is_u_tone(first)) {          /* yu -> ü: yue -> üe */
            pys_add(&r, uv_map(first));
            for (int i = 2; i < p->n; i++) pys_add(&r, p->cp[i]);
        } else if (is_i_tone(first)) {   /* yi -> i */
            for (int i = 1; i < p->n; i++) pys_add(&r, p->cp[i]);
        } else {                          /* ya -> ia */
            pys_add(&r, 'i');
            for (int i = 1; i < p->n; i++) pys_add(&r, p->cp[i]);
        }
        *p = r;
    }
    if (raw.n > 0 && raw.cp[0] == 'w') {
        uint32_t first = (p->n > 1) ? p->cp[1] : 0;
        pys r; pys_clear(&r);
        if (is_u_tone(first)) {           /* wu -> u */
            for (int i = 1; i < p->n; i++) pys_add(&r, p->cp[i]);
        } else {                          /* wa -> ua */
            pys_add(&r, 'u');
            for (int i = 1; i < p->n; i++) pys_add(&r, p->cp[i]);
        }
        *p = r;
    }
    if (!in_finals(p)) *p = raw;
}
static void cf_uv(pys *p) {
    if (p->n >= 2 && (p->cp[0]=='j'||p->cp[0]=='q'||p->cp[0]=='x') && is_u_tone(p->cp[1])) {
        pys r; pys_clear(&r);
        pys_add(&r, p->cp[0]); pys_add(&r, uv_map(p->cp[1]));
        for (int i = 2; i < p->n; i++) pys_add(&r, p->cp[i]);
        *p = r;
    }
}
/* insert 'o' before last codepoint if it ends with ('i',u-tone) and prefix all ascii */
static void cf_iou(pys *p) {
    if (p->n >= 2 && p->cp[p->n-2]=='i' && is_u_tone(p->cp[p->n-1])) {
        int ok = 1;
        for (int i = 0; i < p->n-2; i++) if (!(p->cp[i]>='a' && p->cp[i]<='z')) { ok=0; break; }
        if (ok) { pys r; pys_clear(&r); for (int i=0;i<p->n-1;i++) pys_add(&r,p->cp[i]);
                  pys_add(&r,'o'); pys_add(&r,p->cp[p->n-1]); *p=r; }
    }
}
/* insert 'e' before last codepoint if ends with ('u',i/u-tone) and p[n-3] ascii */
static void cf_uei(pys *p) {
    if (p->n >= 3 && p->cp[p->n-2]=='u' && is_i_tone(p->cp[p->n-1]) &&
        p->cp[p->n-3]>='a' && p->cp[p->n-3]<='z') {
        pys r; pys_clear(&r); for (int i=0;i<p->n-1;i++) pys_add(&r,p->cp[i]);
        pys_add(&r,'e'); pys_add(&r,p->cp[p->n-1]); *p=r;
    }
}
static void cf_uen(pys *p) {
    /* unmarked input: [initial]...un -> ...uen (convert_finals runs post strip_tone) */
    if (p->n >= 3 && p->cp[p->n-2]=='u' && p->cp[p->n-1]=='n' &&
        p->cp[p->n-3]>='a' && p->cp[p->n-3]<='z') {
        pys r; pys_clear(&r); for (int i=0;i<p->n-1;i++) pys_add(&r,p->cp[i]);
        pys_add(&r,'e'); pys_add(&r,p->cp[p->n-1]); *p=r;
    }
}
static void convert_finals(pys *p) {
    cf_zero_consonant(p); cf_uv(p); cf_iou(p); cf_uei(p); cf_uen(p);
}

/* ---------------- initials / finals ---------------- */
static const char *INIT_STRICT[] = {"b","p","m","f","d","t","n","l","g","k","h","j","q","x",
                                    "zh","ch","sh","r","z","c","s"};
static const char *INIT_NOTSTRICT[] = {"b","p","m","f","d","t","n","l","g","k","h","j","q","x",
                                       "zh","ch","sh","r","z","c","s","y","w"};
static const char *get_initials(const pys *p, int strict) {
    const char **list = strict ? INIT_STRICT : INIT_NOTSTRICT;
    int nl = strict ? 21 : 23;
    for (int i = 0; i < nl; i++) if (pys_startswith_ascii(p, list[i])) return list[i];
    return "";
}
static void get_finals(const pys *in, int strict, char *out) {
    pys p = *in;
    if (strict) convert_finals(&p);
    const char *init = get_initials(&p, strict);
    int il = (int)strlen(init);
    pys f; pys_clear(&f);
    for (int i = il; i < p.n; i++) pys_add(&f, p.cp[i]);
    if (strict && !in_finals(&f)) {
        const char *ins = get_initials(&p, 0);
        int isl = (int)strlen(ins);
        pys f2; pys_clear(&f2);
        for (int i = isl; i < p.n; i++) pys_add(&f2, p.cp[i]);
        if (in_finals(&f2)) { pys_to_utf8(&f2, out); return; }
        out[0] = 0; return;
    }
    if (f.n == 0 && !strict) { pys_to_utf8(&p, out); return; }
    pys_to_utf8(&f, out);
}

/* strip tone marks -> no-tone pinyin with ü preserved */
static void strip_tone(const pys *in, pys *out) {
    pys_clear(out);
    for (int i = 0; i < in->n; i++) {
        uint32_t c = in->cp[i];
        if (c >= 0x0300 && c <= 0x036F) continue;   /* drop combining mark */
        pys_add(out, unmark(c));
    }
}
/* first tone digit in a toned pinyin (handles m/ê combining best-effort) */
static int first_tone_digit(const pys *p) {
    for (int i = 0; i < p->n; i++) {
        uint32_t c = p->cp[i];
        int d = tone_digit(c);
        if (d) return d;
        if (i+1 < p->n && ((c=='m' || c==0x00EA) && (p->cp[i+1]==0x0304 || p->cp[i+1]==0x030C || p->cp[i+1]==0x0300))) {
            if (p->cp[i+1]==0x0304) return 1;
            if (p->cp[i+1]==0x030C) return 3;
            if (c=='m' && p->cp[i+1]==0x0300) return 4;
        }
    }
    return 0;
}

/* ---------------- style conversion of a single toned pinyin ---------------- */
static void conv_initials(const pys *toned, char *out) {
    pys_to_utf8(toned, out);
    const char *ini = get_initials(toned, 1);
    strncpy(out, ini, 24); out[24] = 0;
}
/* v_to_u=False (G2P default): render ü as 'v' in a UTF-8 string, in place. */
static void uv_to_v(char *s) {
    char *r = s;
    while (*s) {
        if ((unsigned char)*s == 0xC3 && (unsigned char)s[1] == 0xBC) { *r++ = 'v'; s += 2; }
        else *r++ = *s++;
    }
    *r = 0;
}
static void conv_finals3(const pys *toned, char *out) {
    /* pinyin = toned minus '5' */
    pys p; pys_clear(&p);
    for (int i = 0; i < toned->n; i++) if (toned->cp[i] != '5') pys_add(&p, toned->cp[i]);
    /* new_pinyin = strip_tone(p)  (ü preserved) */
    pys np; strip_tone(&p, &np);
    char finals[24];
    get_finals(&np, 1, finals);
    uv_to_v(finals);
    if (!finals[0]) { out[0] = 0; return; }
    int d = first_tone_digit(&p);
    char res[32];
    if (d) { snprintf(res, sizeof(res), "%s%d", finals, d); }
    else   { snprintf(res, sizeof(res), "%s", finals); }
    /* neutral post: non-empty and no digit -> append '5' */
    int has_digit = 0; for (const char *q=res; *q; q++) if (*q>='0'&&*q<='9') { has_digit=1; break; }
    if (!has_digit) { size_t L=strlen(res); res[L]='5'; res[L+1]=0; }
    strcpy(out, res);
}

/* ---------------- file + hash helpers ---------------- */
static int read_whole(const char *path, unsigned char **out, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return -1; }
    fclose(f); b[n] = 0; *out = b; *out_n = (size_t)n; return 0;
}
static uint64_t fnv(const unsigned char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ULL; }
    return h;
}
static size_t next_pow2(size_t v) { size_t p = 1; while (p < v) p <<= 1; return p; }

/* phrase table lookup: returns data_off + sets nc, or -1 */
static int ph_lookup(gsv_g2p_zh *g, const unsigned char *s, uint32_t n, uint16_t *nc) {
    if (!g->ph_tab_sz) return -1;
    uint64_t h = fnv(s, n);
    size_t i = h & (g->ph_tab_sz - 1);
    for (size_t probe = 0; probe < g->ph_tab_sz; probe++, i = (i + 1) & (g->ph_tab_sz - 1)) {
        if (!g->ph_tab[i].k) return -1;
        if (g->ph_tab[i].kl == n && memcmp(g->ph_tab[i].k, s, n) == 0) { *nc = g->ph_tab[i].nc; return (int)g->ph_tab[i].off; }
    }
    return -1;
}
static void ph_insert(gsv_g2p_zh *g, const unsigned char *s, uint32_t n, uint32_t off, uint16_t nc) {
    uint64_t h = fnv(s, n);
    size_t i = h & (g->ph_tab_sz - 1);
    for (size_t probe = 0; probe < g->ph_tab_sz; probe++, i = (i + 1) & (g->ph_tab_sz - 1)) {
        if (!g->ph_tab[i].k) {
            g->ph_tab[i].k = (const char *)s; g->ph_tab[i].kl = n; g->ph_tab[i].off = off; g->ph_tab[i].nc = nc;
            return;
        }
    }
}
static int pre_lookup(gsv_g2p_zh *g, const unsigned char *s, uint32_t n) {
    if (!g->pre_tab_sz) return 0;
    uint64_t h = fnv(s, n);
    size_t i = h & (g->pre_tab_sz - 1);
    for (size_t probe = 0; probe < g->pre_tab_sz; probe++, i = (i + 1) & (g->pre_tab_sz - 1)) {
        if (!g->pre_tab[i].p) return 0;
        if (g->pre_tab[i].l == n && memcmp(g->pre_tab[i].p, s, n) == 0) return 1;
    }
    return 0;
}
static void pre_insert(gsv_g2p_zh *g, const unsigned char *s, uint32_t n) {
    uint64_t h = fnv(s, n);
    size_t i = h & (g->pre_tab_sz - 1);
    for (size_t probe = 0; probe < g->pre_tab_sz; probe++, i = (i + 1) & (g->pre_tab_sz - 1)) {
        if (!g->pre_tab[i].p) { g->pre_tab[i].p = (const char *)s; g->pre_tab[i].l = n; return; }
    }
}

/* intern a byte string into g->intern, returns pointer */
static const char *intern_bytes(gsv_g2p_zh *g, const unsigned char *s, size_t n) {
    if (gsv_buf_reserve(&g->intern, n + 1) != 0) return NULL;   /* ensure capacity first */
    char *p = g->intern.data + g->intern.len;                   /* data is now stable */
    memcpy(g->intern.data + g->intern.len, s, n); g->intern.len += n;
    g->intern.data[g->intern.len] = 0; g->intern.len += 1;
    return p;
}

/* ---------------- loaders ---------------- */
static int load_pinyin(gsv_g2p_zh *g, const char *dir) {
    char path[1024]; snprintf(path, sizeof(path), "%s/pinyin.bin", dir);
    unsigned char *b; size_t n;
    if (read_whole(path, &b, &n) != 0) return -1;
    if (n < 12 || memcmp(b, "GSPY", 4) != 0) { free(b); return -1; }
    uint32_t count; memcpy(&count, b + 8, 4);
    g->py_cp = (uint32_t *)malloc(sizeof(uint32_t) * count);
    g->py_val = (char **)malloc(sizeof(char *) * count);
    /* reserve intern arena up-front so stored pointers never dangle on realloc */
    size_t total_val = 0, off = 12;
    for (uint32_t i = 0; i < count; i++) {
        off += 4; uint16_t ln; memcpy(&ln, b + off, 2); off += 2 + ln; total_val += ln;
    }
    gsv_buf_reserve(&g->intern, total_val + (size_t)count + 64);
    off = 12;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t cp; uint16_t ln;
        memcpy(&cp, b + off, 4); off += 4;
        memcpy(&ln, b + off, 2); off += 2;
        g->py_cp[i] = cp;
        /* lazy_pinyin uses only the first candidate (pys[0]) */
        size_t fl = ln; for (size_t k = 0; k < ln; k++) if (b[off + k] == ',') { fl = k; break; }
        g->py_val[i] = (char *)intern_bytes(g, b + off, fl);
        off += ln;
    }
    g->n_py = (int)count;
    free(b);
    return 0;
}

static int load_phrases(gsv_g2p_zh *g, const char *dir) {
    char path[1024]; snprintf(path, sizeof(path), "%s/phrases.bin", dir);
    unsigned char *b; size_t n;
    if (read_whole(path, &b, &n) != 0) return -1;
    if (n < 12 || memcmp(b, "GSPH", 4) != 0) { free(b); return -1; }
    uint32_t count; memcpy(&count, b + 8, 4);

    g->ph_tab_sz = next_pow2((size_t)count * 2 + 16);
    g->ph_tab = (gsv_ph_slot *)calloc(g->ph_tab_sz, sizeof(gsv_ph_slot));
    size_t prefix_total = 0, sum_plen = 0, sum_prelen = 0;
    /* first pass: compute phrase data + prefix count */
    size_t off = 12;
    uint32_t *poff = (uint32_t *)malloc(sizeof(uint32_t) * count);
    uint16_t *pnc  = (uint16_t *)malloc(sizeof(uint16_t) * count);
    for (uint32_t i = 0; i < count; i++) {
        uint16_t plen, nchars;
        memcpy(&plen, b + off, 2); off += 2;
        off += plen;
        memcpy(&nchars, b + off, 2); off += 2;
        poff[i] = (uint32_t)g->ph_data_holder.len;
        pnc[i] = nchars;
        for (uint16_t c = 0; c < nchars; c++) {
            uint8_t ncands; memcpy(&ncands, b + off, 1); off += 1;
            /* store only first candidate */
            uint16_t clen; memcpy(&clen, b + off, 2); off += 2;
            const unsigned char *cand = b + off; off += clen;
            gsv_buf_append(&g->ph_data_holder, cand, clen);
            gsv_buf_append(&g->ph_data_holder, (const void *)"", 1);
            for (uint8_t k = 1; k < ncands; k++) { uint16_t l2; memcpy(&l2, b+off,2); off+=2+l2; }
        }
        prefix_total += plen;   /* all prefixes of this phrase */
        sum_plen  += plen;                              /* key bytes */
        sum_prelen += (size_t)plen * (plen + 1) / 2;    /* 1+..+plen prefix bytes */
    }
    g->ph_data = g->ph_data_holder.data;

    /* build prefix set */
    g->pre_tab_sz = next_pow2(prefix_total * 2 + 16);
    g->pre_tab = (gsv_pre_slot *)calloc(g->pre_tab_sz, sizeof(gsv_pre_slot));
    off = 12;
    for (uint32_t i = 0; i < count; i++) {
        uint16_t plen, nchars;
        memcpy(&plen, b + off, 2); off += 2;
        const unsigned char *ph = b + off; off += plen;
        memcpy(&nchars, b + off, 2); off += 2;
        /* intern phrase key + all its prefixes (b is freed at end) */
        const char *key = intern_bytes(g, ph, plen);
        ph_insert(g, (const unsigned char *)key, plen, poff[i], nchars);
        for (uint16_t k = 1; k <= plen; k++) {
            const char *pk = intern_bytes(g, ph, k);
            pre_insert(g, (const unsigned char *)pk, k);
        }
        /* skip per-char data */
        for (uint16_t c = 0; c < nchars; c++) {
            uint8_t ncands; memcpy(&ncands, b + off, 1); off += 1;
            for (uint8_t k = 0; k < ncands; k++) { uint16_t l2; memcpy(&l2, b+off,2); off+=2+l2; }
        }
    }
    free(poff); free(pnc); free(b);
    return 0;
}

/* Pre-scan both data files to compute the exact intern arena size needed, so a
 * single up-front reserve prevents any realloc from dangling stored pointers. */
static size_t gsv_scan_intern_total(const char *dir) {
    unsigned char *b; size_t n; char path[1024];
    size_t val = 0, keys = 0, pref = 0;
    snprintf(path, sizeof(path), "%s/pinyin.bin", dir);
    if (read_whole(path, &b, &n) != 0) return 0;
    uint32_t count; memcpy(&count, b + 8, 4);
    size_t off = 12;
    for (uint32_t i = 0; i < count; i++) { off += 4; uint16_t ln; memcpy(&ln, b + off, 2); off += 2 + ln; val += ln; }
    free(b);
    snprintf(path, sizeof(path), "%s/phrases.bin", dir);
    if (read_whole(path, &b, &n) != 0) return 0;
    uint32_t pc; memcpy(&pc, b + 8, 4);
    off = 12;
    for (uint32_t i = 0; i < pc; i++) {
        uint16_t plen, nch; memcpy(&plen, b + off, 2); off += 2; off += plen;
        memcpy(&nch, b + off, 2); off += 2;
        keys += plen; pref += (size_t)plen * (plen + 1) / 2;
        for (uint16_t c = 0; c < nch; c++) { uint8_t nc; memcpy(&nc, b + off, 1); off += 1;
            for (uint8_t k = 0; k < nc; k++) { uint16_t l2; memcpy(&l2, b + off, 2); off += 2 + l2; } }
    }
    free(b);
    return val + keys + pref;
}

int gsv_g2p_zh_load(gsv_g2p_zh *g, const char *dir) {
    memset(g, 0, sizeof(*g));
    /* reserve the whole intern arena up-front (data + NULs + margin) */
    size_t tot = gsv_scan_intern_total(dir);
    gsv_buf_reserve(&g->intern, tot + 1024 * 1024);
    if (load_pinyin(g, dir) != 0) return -1;
    if (load_phrases(g, dir) != 0) return -1;
    return 0;
}

void gsv_g2p_zh_free(gsv_g2p_zh *g) {
    free(g->py_cp); free(g->py_val);
    free(g->intern.data); free(g->ph_data);
    free(g->ph_tab); free(g->pre_tab);
    free(g->outbuf.data); free(g->tonebuf.data);
    memset(g, 0, sizeof(*g));
}

/* ---------------- mmseg (no_non_phrases=True) per-char toned pinyin ---------------- */
/* fill `toned` with one pointer per char of cps[0..n) via phrase/char lookup. returns n. */
static int mmseg_fill(gsv_g2p_zh *g, const uint32_t *cps, int n, char **toned) {
    int out = 0;
    int pos = 0;
    while (pos < n) {
        /* forward max match: longest prefix of cps[pos..] that is a full phrase */
        int last_valid_end = -1;   /* exclusive end of longest valid phrase */
        int i = pos;
        for (; i < n; i++) {
            char ubuf[64]; int ulen = 0;
            for (int k = pos; k <= i; k++) ulen += gsv_utf8_encode(cps[k], ubuf + ulen);
            if (!pre_lookup(g, (const unsigned char *)ubuf, (uint32_t)ulen)) break;
            uint16_t nc;
            if (ph_lookup(g, (const unsigned char *)ubuf, (uint32_t)ulen, &nc) >= 0)
                last_valid_end = i + 1;
        }
        int seg_end;
        if (last_valid_end > 0) {
            seg_end = last_valid_end;
        } else {
            /* prefix matched but no full phrase within, or no prefix: first char only */
            seg_end = pos + 1;
        }
        /* emit segment cps[pos..seg_end) */
        int slen = seg_end - pos;
        char ubuf[64]; int ulen = 0;
        for (int k = pos; k < seg_end; k++) ulen += gsv_utf8_encode(cps[k], ubuf + ulen);
        uint16_t nc;
        int data_off = ph_lookup(g, (const unsigned char *)ubuf, (uint32_t)ulen, &nc);
        if (data_off >= 0) {
            /* phrase: per-char first candidate */
            const unsigned char *d = (const unsigned char *)g->ph_data + data_off;
            for (int c = 0; c < slen; c++) { toned[out++] = (char *)d; d += strlen((char *)d) + 1; }
        } else {
            /* per-char PINYIN_DICT lookup */
            for (int c = pos; c < seg_end; c++) {
                uint32_t cp = cps[c];
                int lo = 0, hi = g->n_py - 1, found = -1;
                while (lo <= hi) { int mid = (lo + hi) >> 1;
                    if (g->py_cp[mid] == cp) { found = mid; break; }
                    else if (g->py_cp[mid] < cp) lo = mid + 1; else hi = mid - 1; }
                if (found >= 0) toned[out++] = g->py_val[found];
                else { char cb[8]; int cl = gsv_utf8_encode(cp, cb);
                       toned[out++] = gsv_buf_intern(&g->tonebuf, cb, (size_t)cl); }
            }
        }
        pos = seg_end;
    }
    return out;
}

/* ---------------- public entry ---------------- */
int gsv_pinyin_lazy(gsv_g2p_zh *g, const char *word, int style, char **out_strs, int *out_n) {
    g->outbuf.len = 0;
    uint32_t cps[1024]; int n = 0;
    const char *p = word; uint32_t c;
    while ((c = gsv_utf8_decode(&p)) != 0 && n < 1024) cps[n++] = c;
    /* per-call scratch for fallback chars; reserved up-front so no realloc invalidates */
    g->tonebuf.len = 0;
    gsv_buf_reserve(&g->tonebuf, (size_t)n * 6 + 16);

    char *toned[1024]; int tn = 0;
    unsigned char han_entry[1024];   /* 1 if this entry came from a han char (convert), 0 if non-han (keep as-is) */
    int i = 0;
    while (i < n) {
        int ishan = gsv_is_han(cps[i]);
        int j = i;
        while (j < n && gsv_is_han(cps[j]) == ishan) j++;
        if (ishan) {
            int before = tn;
            tn += mmseg_fill(g, cps + i, j - i, toned + tn);
            for (int k = before; k < tn; k++) han_entry[k] = 1;
        } else {
            /* non-han block: one entry = whole block (kept as-is) */
            char ubuf[1024]; int ulen = 0;
            for (int k = i; k < j; k++) ulen += gsv_utf8_encode(cps[k], ubuf + ulen);
            toned[tn] = gsv_buf_intern(&g->tonebuf, ubuf, (size_t)ulen);
            han_entry[tn] = 0; tn++;
        }
        i = j;
    }

    for (int k = 0; k < tn; k++) {
        if (!han_entry[k]) { out_strs[k] = gsv_buf_intern(&g->outbuf, toned[k], strlen(toned[k])); continue; }
        pys tp; pys_from_utf8(&tp, toned[k]);
        char res[32];
        if (style == GSV_PY_STYLE_INITIALS) conv_initials(&tp, res);
        else conv_finals3(&tp, res);
        out_strs[k] = gsv_buf_intern(&g->outbuf, res, strlen(res));
    }
    *out_n = tn;
    return 0;
}
