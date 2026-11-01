/* fft.h - radix-2 FFT (in-place, complex float32) */
#ifndef GSV_FFT_H
#define GSV_FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Complex as interleaved float32: re[0],im[0],re[1],im[1],... */
typedef struct {
    int n;              /* power of two */
    float *twiddle;     /* [n/2] e^{-2pi i k/n} interleaved, cached */
} gsv_fft_plan;

/* Returns 0 on success. n must be a power of two >= 2.
 * Plan may be reused across calls (thread: not thread-safe per plan). */
int gsv_fft_plan_create(gsv_fft_plan *plan, int n);
void gsv_fft_plan_destroy(gsv_fft_plan *plan);

/* In-place FFT. sign=+1 forward, -1 inverse (no 1/n normalization). */
void gsv_fft_execute(const gsv_fft_plan *plan, float *reim, int sign);

/* rfft for real input: fills spec[0..n/2] amplitudes from real signal x[n].
 * Uses the plan's forward FFT. */
void gsv_rfft_amplitude(const gsv_fft_plan *plan, const float *x, float *spec /* [n/2+1] */);

#ifdef __cplusplus
}
#endif

#endif
