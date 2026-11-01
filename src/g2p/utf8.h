/* utf8.h - minimal UTF-8 + growable string helpers for the G2P port. */
#ifndef GSV_G2P_UTF8_H
#define GSV_G2P_UTF8_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

/* Decode one codepoint at *s, advancing *s past it. Returns 0 on invalid/EOF. */
static inline uint32_t gsv_utf8_decode(const char **s) {
    unsigned char c = (unsigned char)(*s)[0];
    if (c < 0x80) { (*s)++; return c; }
    if ((c & 0xE0) == 0xC0) {
        const char *p = *s;
        unsigned char c2 = (unsigned char)p[1];
        if ((c2 & 0xC0) != 0x80) return 0;
        uint32_t cp = ((uint32_t)(c & 0x1F) << 6) | (c2 & 0x3F);
        *s = p + 2; return cp;
    }
    if ((c & 0xF0) == 0xE0) {
        const char *p = *s;
        unsigned char c2 = (unsigned char)p[1], c3 = (unsigned char)p[2];
        if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return 0;
        uint32_t cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(c2 & 0x3F) << 6) | (c3 & 0x3F);
        *s = p + 3; return cp;
    }
    if ((c & 0xF8) == 0xF0) {
        const char *p = *s;
        unsigned char c2 = (unsigned char)p[1], c3 = (unsigned char)p[2], c4 = (unsigned char)p[3];
        if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80 || (c4 & 0xC0) != 0x80) return 0;
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(c2 & 0x3F) << 12) |
                      ((uint32_t)(c3 & 0x3F) << 6) | (c4 & 0x3F);
        *s = p + 4; return cp;
    }
    return 0;
}

/* Encode codepoint into buf (needs up to 4 bytes). Returns bytes written. */
static inline int gsv_utf8_encode(uint32_t cp, char *buf) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F)); return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

/* Number of codepoints in a UTF-8 string. */
static inline int gsv_utf8_len(const char *s) {
    int n = 0; const char *p = s; uint32_t cp;
    while ((cp = gsv_utf8_decode(&p)) != 0) n++;
    return n;
}

/* pypinyin RE_HANS (SUPPORT_UCS4 form): is this a han char with pinyin? */
static inline int gsv_is_han(uint32_t cp) {
    if (cp == 0x3007) return 1;
    if (cp >= 0xE815 && cp <= 0xE864) return 1;
    if (cp == 0xFA18) return 1;
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 1;
    if (cp >= 0x2A703 && cp <= 0x2B73F) return 1;
    if (cp >= 0x2B740 && cp <= 0x2B81D) return 1;
    if (cp >= 0x2B825 && cp <= 0x2BF6E) return 1;
    if (cp >= 0x2C029 && cp <= 0x2CE93) return 1;
    if (cp == 0x2D016) return 1;
    if (cp >= 0x2D11B && cp <= 0x2EBD9) return 1;
    if (cp >= 0x2F80A && cp <= 0x2FA1F) return 1;
    if (cp >= 0x30000 && cp <= 0x3134A) return 1;
    if (cp >= 0x300F7 && cp <= 0x31288) return 1;
    if (cp == 0x30EDD || cp == 0x30EDE) return 1;
    if (cp >= 0x31350 && cp <= 0x32389) return 1;
    return 0;
}

/* jieba re_han_default: [\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-] */
static inline int gsv_is_jieba_han(uint32_t cp) {
    if (cp >= 0x4E00 && cp <= 0x9FD5) return 1;
    unsigned char c = (unsigned char)cp;   /* ascii only beyond CJK */
    if (c >= 'a' && c <= 'z') return 1;
    if (c >= 'A' && c <= 'Z') return 1;
    if (c >= '0' && c <= '9') return 1;
    switch (c) { case '+': case '#': case '&': case '.': case '_': case '%': case '-': return 1; }
    return 0;
}

/* ---- growable byte buffer ---- */
typedef struct gsv_buf {
    char *data;
    size_t len, cap;
} gsv_buf;

static inline int gsv_buf_reserve(gsv_buf *b, size_t need) {
    if (b->len + need <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 64;
    while (nc < b->len + need) nc *= 2;
    char *nd = (char *)realloc(b->data, nc);
    if (!nd) return -1;
    b->data = nd; b->cap = nc;
    return 0;
}

static inline int gsv_buf_append(gsv_buf *b, const void *p, size_t n) {
    if (gsv_buf_reserve(b, n) != 0) return -1;
    memcpy(b->data + b->len, p, n);
    b->len += n;
    return 0;
}

static inline int gsv_buf_append_str(gsv_buf *b, const char *s) {
    return gsv_buf_append(b, s, strlen(s));
}

static inline int gsv_buf_append_char(gsv_buf *b, char c) {
    return gsv_buf_append(b, &c, 1);
}

/* append a NUL-terminated copy of s into buf, returns pointer to it (or NULL). */
static inline char *gsv_buf_intern(gsv_buf *b, const char *s, size_t n) {
    if (gsv_buf_reserve(b, n + 1) != 0) return NULL;   /* ensure capacity first */
    char *p = b->data + b->len;                        /* data is now stable */
    memcpy(b->data + b->len, s, n); b->len += n;
    b->data[b->len] = 0; b->len += 1;
    return p;
}

#endif
