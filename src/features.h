/* features.h - ref warmup feature extraction (ge + prompt codes) */
#ifndef GSV_FEATURES_H
#define GSV_FEATURES_H

#include "engine.h"

/* ge path: wav32k must be peak-normalized (maxx>1 -> /= min(2,maxx));
 * wav16k must be the 32k signal resampled down (per _get_spec).
 * Fills ref->ge [gin] and ref->ge512 [512]. */
int gsv_ref_compute_ge(gsv_engine *e, const gsv_voice_cfg *cfg,
                       const float *wav32k, int64_t n32,
                       const float *wav16k, int64_t n16,
                       gsv_ref *ref);

/* prompt codes path: wav16k = direct resample from source rate with 0.3s
 * silence already appended. Fills ref->prompt_codes / n_prompt_codes. */
int gsv_ref_compute_codes(gsv_engine *e, const float *wav16k, int64_t n16,
                          gsv_ref *ref);

#endif
