/* en_hashmap.h - open-addressing string->uint32 map (FNV-1a 64, linear probing).
 * Used by the English G2P data loaders (dicts, tagger, wordsegment). */
#ifndef GSV_G2P_EN_HASHMAP_H
#define GSV_G2P_EN_HASHMAP_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t *hash;   /* 0 = empty slot */
    uint32_t *val;    /* value (offset / index) */
    uint16_t *klen;
    uint32_t *koff;   /* key offset into keybuf */
    char     *keybuf;
    size_t    keylen; /* bytes used in keybuf */
    size_t    keycap; /* allocated bytes in keybuf */
    int cap, mask, n;
} gsv_strmap;

static inline int gsv_buf_reserve_key(gsv_strmap *m, size_t need) {
    if (m->keylen + need <= m->keycap) return 0;
    size_t nc = m->keycap ? m->keycap : 256;
    while (nc < m->keylen + need) nc *= 2;
    char *nd = (char *)realloc(m->keybuf, nc);
    if (!nd) return -1;
    m->keybuf = nd; m->keycap = nc;
    return 0;
}

static inline uint64_t gsv_fnv1a(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static inline int gsv_strmap_init(gsv_strmap *m, int expected) {
    int cap = 256;
    while (cap < expected * 2 + 16) cap <<= 1;   /* load factor <= ~0.5 */
    m->hash = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    m->val  = (uint32_t *)calloc((size_t)cap, sizeof(uint32_t));
    m->klen = (uint16_t *)calloc((size_t)cap, sizeof(uint16_t));
    m->koff = (uint32_t *)calloc((size_t)cap, sizeof(uint32_t));
    if (!m->hash || !m->val || !m->klen || !m->koff) return -1;
    m->cap = cap; m->mask = cap - 1; m->n = 0;
    return 0;
}

static inline void gsv_strmap_free(gsv_strmap *m) {
    free(m->hash); free(m->val); free(m->klen); free(m->koff); free(m->keybuf);
    memset(m, 0, sizeof *m);
}

/* insert key (last wins on duplicate). Returns 0 ok. */
static inline int gsv_strmap_put(gsv_strmap *m, const void *key, int len, uint32_t val) {
    if (len < 0 || len > 65535) return -1;
    uint64_t h = gsv_fnv1a(key, (size_t)len);
    size_t i = (size_t)(uint32_t)h & (size_t)m->mask;
    int inserted = 0;
    for (;;) {
        if (m->hash[i] == 0) {
            if (gsv_buf_reserve_key(m, (size_t)len) != 0) return -1;
            m->koff[i] = (uint32_t)m->keylen;
            memcpy(m->keybuf + m->keylen, key, (size_t)len);
            m->keylen += (size_t)len;
            m->hash[i] = h;
            m->val[i] = val;
            m->klen[i] = (uint16_t)len;
            m->n++;
            inserted = 1;
            break;
        }
        if (m->hash[i] == h && m->klen[i] == (uint16_t)len &&
            memcmp(m->keybuf + m->koff[i], key, (size_t)len) == 0) {
            m->val[i] = val;   /* duplicate: last wins */
            return 0;
        }
        i = (i + 1) & (size_t)m->mask;
    }
    (void)inserted;
    return 0;
}

/* lookup: returns value or UINT32_MAX if absent. */
static inline uint32_t gsv_strmap_get(const gsv_strmap *m, const void *key, int len) {
    if (len < 0 || len > 65535) return UINT32_MAX;
    uint64_t h = gsv_fnv1a(key, (size_t)len);
    size_t i = (size_t)(uint32_t)h & (size_t)m->mask;
    for (;;) {
        if (m->hash[i] == 0) return UINT32_MAX;
        if (m->hash[i] == h && m->klen[i] == (uint16_t)len &&
            memcmp(m->keybuf + m->koff[i], key, (size_t)len) == 0)
            return m->val[i];
        i = (i + 1) & (size_t)m->mask;
    }
}

#endif
