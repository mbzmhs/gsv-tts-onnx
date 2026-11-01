/* en_dicts.c - loader for cmu.bin / namedict.bin (word -> phoneme-index list).
 * The strmap's keybuf holds the word bytes; pron indices live in d->prons. */
#include "en_dicts.h"
#include "en_util.h"

int gsv_en_dict_load(gsv_en_dict *d, const char *dir, const char *name, uint32_t magic) {
    memset(d, 0, sizeof *d);
    size_t len = 0;
    char *buf = gsv_en_read_file(dir, name, &len);
    if (!buf || len < 8) { free(buf); return -1; }
    if (rd_u32(buf) != magic) { free(buf); return -2; }
    int n = (int)rd_u32(buf + 4);
    const char *p = buf + 8;

    d->n = n;
    d->poff = (uint32_t *)calloc((size_t)(n > 0 ? n : 1), sizeof(uint32_t));
    d->plen = (uint8_t *)calloc((size_t)(n > 0 ? n : 1), 1);
    if (!d->poff || !d->plen) return -3;
    if (gsv_strmap_init(&d->map, n) != 0) return -4;

    size_t pcap = 256;
    d->prons = (uint8_t *)malloc(pcap);
    if (!d->prons) return -5;
    size_t pronlen = 0;

    for (int i = 0; i < n; i++) {
        int wlen = (unsigned char)p[0]; p++;
        if ((size_t)(p - buf) + (size_t)wlen > len) return -6;
        const char *wordp = p;
        int plen = (unsigned char)p[wlen];
        p += wlen + 1;
        if ((size_t)(p - buf) + (size_t)plen > len) return -7;

        d->plen[i] = (uint8_t)plen;
        if (pronlen + (size_t)plen > pcap) {
            while (pcap < pronlen + (size_t)plen) pcap *= 2;
            uint8_t *nd = (uint8_t *)realloc(d->prons, pcap); if (!nd) return -8;
            d->prons = nd;
        }
        d->poff[i] = (uint32_t)pronlen;
        memcpy(d->prons + pronlen, p, (size_t)plen);
        pronlen += (size_t)plen;
        p += plen;

        if (gsv_strmap_put(&d->map, wordp, wlen, (uint32_t)i) != 0) return -9;
    }
    free(buf);
    return 0;
}

void gsv_en_dict_free(gsv_en_dict *d) {
    gsv_strmap_free(&d->map);
    free(d->poff); free(d->plen); free(d->prons);
    memset(d, 0, sizeof *d);
}

const uint8_t *gsv_en_dict_get(const gsv_en_dict *d, const char *word, int *n) {
    uint32_t i = gsv_strmap_get(&d->map, word, (int)strlen(word));
    if (i == UINT32_MAX || i >= (uint32_t)d->n) return NULL;
    if (n) *n = d->plen[i];
    return d->prons + d->poff[i];
}
