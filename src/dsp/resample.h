/* resample.h - bandlimited FIR resampler, exact port of
 * torchaudio.functional.resample(method="sinc_interp_hann", width=6, rolloff=0.99) */
#ifndef GSV_RESAMPLE_H
#define GSV_RESAMPLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int sr0, sr1;      /* original / target sample rate */
    int o, n;          /* reduced ratio (sr/gcd) */
    int width;         /* lowpass_filter_width * sr0/base, ceil */
    int klen;          /* 2*width + o */
    float *kernels;    /* [n][klen] row-major */
} gsv_resampler;

/* Returns 0 on success. */
int gsv_resampler_create(gsv_resampler *r, int sr0, int sr1);
void gsv_resampler_destroy(gsv_resampler *r);

/* Output length for input of n samples: ceil(n * sr1 / sr0) */
int64_t gsv_resampler_out_len(const gsv_resampler *r, int64_t n);

/* y = resample(x). x freed by caller. Returns malloc'd float array (caller frees). */
float *gsv_resample(const gsv_resampler *r, const float *x, int64_t n, int64_t *out_n);

#ifdef __cplusplus
}
#endif

#endif
