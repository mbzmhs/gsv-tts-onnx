/* en_textnorm.h - C port of G2P/English/Normalization/expend.py normalize()
 * (incl. the inflect number_to_words/ordinal subset it uses). */
#ifndef GSV_G2P_EN_TEXTNORM_H
#define GSV_G2P_EN_TEXTNORM_H

#include "utf8.h"

/* Full expend.normalize(text). Result NUL-terminated in out (caller frees out->data). */
int gsv_en_textnorm_normalize(const char *text, gsv_buf *out);

#endif
