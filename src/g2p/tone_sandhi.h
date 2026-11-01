/* tone_sandhi.h - C port of GPT_SoVITS/G2P/Chinese/tone_sandhi.py (ToneSandhi).
 * Provides pre_merge_for_modify and modified_tone, the two entry points chinese.py uses. */
#ifndef GSV_TONE_SANDHI_H
#define GSV_TONE_SANDHI_H

#include "jieba.h"
#include "pypinyin.h"
#include "utf8.h"

/* One (word, pos) segment. w/p are NUL-terminated UTF-8; after pre_merge they may
 * point into the arena passed to gsv_ts_pre_merge (keep it alive while using them). */
typedef struct { const char *w; const char *p; } gsv_ts_seg;

/* pre_merge_for_modify: in[0..n-1] -> out[0..*out_n-1]. Merged strings are stored
 * in arena `ar` (a gsv_buf); keep `ar` alive while using the returned segs.
 * Returns the number of output segments. */
int gsv_ts_pre_merge(gsv_jieba *g, gsv_g2p_zh *py, gsv_buf *ar,
                     const gsv_ts_seg *in, int n, gsv_ts_seg *out, int maxout);

/* modified_tone: applies bu/yi/neural/three sandhi to finals[0..n-1] IN PLACE.
 * word/pos are UTF-8; finals[i] must be writable (each ends with a tone digit). */
void gsv_ts_modified_tone(gsv_jieba *g, const char *word, const char *pos, char **finals, int n);

#endif
