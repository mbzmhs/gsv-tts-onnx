/* en_dicts.h - loader for cmu.bin / namedict.bin (word -> phoneme-index list). */
#ifndef GSV_G2P_EN_DICTS_H
#define GSV_G2P_EN_DICTS_H

#include <stdint.h>
#include "en_hashmap.h"

typedef struct {
    gsv_strmap map;      /* word -> entry index (keybuf holds the word bytes) */
    uint32_t  *poff;     /* per-entry: pron offset into prons */
    uint8_t   *plen;
    uint8_t   *prons;    /* concatenated phoneme-index bytes */
    int n;
} gsv_en_dict;

/* magic: 0x4D444331 (cmu) or 0x4D444E31 (namedict). Returns 0 ok. */
int  gsv_en_dict_load(gsv_en_dict *d, const char *dir, const char *name, uint32_t magic);
void gsv_en_dict_free(gsv_en_dict *d);

/* Returns pointer into d->prons and sets *n, or NULL if absent. */
const uint8_t *gsv_en_dict_get(const gsv_en_dict *d, const char *word, int *n);

#endif
