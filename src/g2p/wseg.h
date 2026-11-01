/* wseg.h - C port of the `wordsegment` package (Norvig-style DP segmentation).
 * Loads wseg_uni.bin / wseg_bi.bin and reproduces Segmenter.segment() exactly:
 * same scores (stupid backoff), same log10 accumulation, same tie-breaking
 * (first/shortest prefix wins), same 250-char chunking with 5-word carry. */
#ifndef GSV_G2P_WSEG_H
#define GSV_G2P_WSEG_H

#include <stdint.h>
#include "en_hashmap.h"

typedef struct {
    gsv_strmap uni;   /* word -> index into unival[] */
    double    *unival;
    int        n_uni;
    gsv_strmap bi;    /* "prev word" -> index into bival[] */
    double    *bival;
    int        n_bi;
} gsv_wseg;

int  gsv_wseg_load(gsv_wseg *w, const char *dir);
void gsv_wseg_free(gsv_wseg *w);

/* segment(text): returns words in out (array of pointers into a malloc'd buffer
 * owned by the call - freed on next gsv_wseg_segment call or by w). n_words set.
 * Mirrors wordsegment.segment(): input is cleaned (lowercased, [a-z0-9] kept). */
const char **gsv_wseg_segment(const gsv_wseg *w, const char *text, int *n_words);

#endif
