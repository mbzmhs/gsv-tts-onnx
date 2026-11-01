/* postproc.h - server-side audio post-processing chain (port of TTS.py). */
#ifndef GSV_POSTPROC_H
#define GSV_POSTPROC_H

#include "gsvtts.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Full post-decode chain, in order:
 * 1. head trim: first frame (len 512, hop 256) with RMS > 0.02 within the first
 *    64000 samples; offset = frame*hop - 3200 (clamped to 0). If no frame
 *    exceeds, drop the whole search window.
 * 2. despike: zero isolated samples with |x| > 0.3 whose +/-160-sample
 *    neighborhood mean (excluding self, replicate edges) < 0.05. Skipped if
 *    length < 321.
 * 3. normalize: divide by max_abs if it exceeds 1.
 * 4. tail pad: append round(0.2 * sr) zero samples.
 * out may alias in; must hold the final length. On success returns 0 and sets
 * *out_len; if out_capacity is too small returns -1 with *out_len = required. */
GSVTTS_API int gsv_audio_postproc(const float *in, int n_in, int sr,
                                  float *out, int out_capacity, int *out_len);

#ifdef __cplusplus
}
#endif

#endif
