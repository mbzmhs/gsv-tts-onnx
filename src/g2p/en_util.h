/* en_util.h - small shared helpers for the English G2P modules. */
#ifndef GSV_G2P_EN_UTIL_H
#define GSV_G2P_EN_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Read a whole file. Tries "dir\\name" then "dir/name". Returns malloc'd NUL-terminated
 * buffer or NULL. */
static char *gsv_en_read_file(const char *dir, const char *name, size_t *out_len) {
    char path[1200];
    snprintf(path, sizeof path, "%s\\%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(path, sizeof path, "%s/%s", dir, name);
        f = fopen(path, "rb");
    }
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = 0; fclose(f);
    if (out_len) *out_len = rd;
    return buf;
}

/* little-endian readers over a byte buffer */
static inline uint32_t rd_u32(const char *p) {
    return (uint32_t)(unsigned char)p[0] | ((uint32_t)(unsigned char)p[1] << 8) |
           ((uint32_t)(unsigned char)p[2] << 16) | ((uint32_t)(unsigned char)p[3] << 24);
}
static inline uint16_t rd_u16(const char *p) {
    return (uint16_t)((uint16_t)(unsigned char)p[0] | ((uint16_t)(unsigned char)p[1] << 8));
}
static inline double rd_f64(const char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | (unsigned char)p[i];
    double d; memcpy(&d, &v, 8); return d;
}

#endif
