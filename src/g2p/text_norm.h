/* text_norm.h - C port of GPT_SoVITS/G2P/Chinese/Normalization/text_normlization.py
 * (TextNormalizer). Exposes normalize(text) = "".join of per-sentence normalization. */
#ifndef GSV_TEXT_NORM_H
#define GSV_TEXT_NORM_H

#include "utf8.h"

/* TextNormalizer.normalize(text): split into sentences, normalize each, return the
 * joined result. Writes a NUL-terminated string into `out` (caller owns/resets it).
 * Returns pointer into out. */
const char *gsv_textnorm_normalize(const char *text, gsv_buf *out);

#endif
