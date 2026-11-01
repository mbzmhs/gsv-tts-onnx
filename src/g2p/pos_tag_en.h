/* pos_tag_en.h - C port of NLTK PerceptronTagger (averaged_perceptron_tagger_eng).
 * Loads pos_tag.bin (classes / tagdict / weights) and reproduces tag() exactly:
 * tagdict fast path, identical feature strings, identical double accumulation
 * order, (score, label) max tie-break. */
#ifndef GSV_G2P_POS_TAG_EN_H
#define GSV_G2P_POS_TAG_EN_H

#include <stdint.h>
#include "en_hashmap.h"

typedef struct {
    char     **classes; int n_classes;   /* tag strings (file order) */
    gsv_strmap tagdict;                  /* word -> tag index */
    gsv_strmap weights;                  /* feature string -> feature index */
    uint16_t  *f_nent;                   /* per feature: entry count */
    uint32_t  *f_eoff;                   /* per feature: entry offset into f_entries */
    uint8_t   *f_enttag;                 /* entries: tag index */
    double    *f_entwgt;                 /* entries: weight (file order preserved) */
    int n_feat;
} gsv_pos_tagger;

int  gsv_pos_tag_load(gsv_pos_tagger *t, const char *dir);
void gsv_pos_tag_free(gsv_pos_tagger *t);

/* Tag tokens. tags_out must hold >= n_words pointers (point into t->classes). */
void gsv_pos_tag(const gsv_pos_tagger *t, const char **words, int n_words, const char **tags_out);

#endif
