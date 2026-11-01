/* fft.c - iterative radix-2 FFT with cached twiddles */
#include "fft.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

int gsv_fft_plan_create(gsv_fft_plan *plan, int n) {
    if (!is_pow2(n) || n < 2) return -1;
    plan->n = n;
    plan->twiddle = (float *)malloc(sizeof(float) * n); /* re,im interleaved: n/2 pairs */
    if (!plan->twiddle) return -1;
    for (int k = 0; k < n / 2; k++) {
        double ang = -2.0 * M_PI * k / n;
        plan->twiddle[2 * k] = (float)cos(ang);
        plan->twiddle[2 * k + 1] = (float)sin(ang);
    }
    return 0;
}

void gsv_fft_plan_destroy(gsv_fft_plan *plan) {
    if (plan->twiddle) free(plan->twiddle);
    plan->twiddle = NULL;
    plan->n = 0;
}

/* bit-reversal permutation in place */
static void bitrev(float *reim, int n) {
    int bits = 0, t;
    for (t = n; t > 1; t >>= 1) bits++;
    for (int i = 0; i < n; i++) {
        int r = 0;
        for (int j = 0; j < bits; j++)
            if (i & (1 << j)) r |= 1 << (bits - 1 - j);
        if (r > i) {
            float tr = reim[2 * i], ti = reim[2 * i + 1];
            reim[2 * i] = reim[2 * r]; reim[2 * i + 1] = reim[2 * r + 1];
            reim[2 * r] = tr; reim[2 * r + 1] = ti;
        }
    }
}

void gsv_fft_execute(const gsv_fft_plan *plan, float *reim, int sign) {
    int n = plan->n;
    bitrev(reim, n);
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        /* twiddle stride: we need e^{-2pi i * sign * k / len}; plan has e^{-2pi i k/n}.
         * index into plan twiddle = k * (n/len). */
        int stride = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                int ti = (k * stride) * 2;
                float wr = plan->twiddle[ti];
                float wi = plan->twiddle[ti + 1] * sign; /* conjugate for inverse */
                int a = i + k, b = a + half;
                float xr = reim[2 * b] * wr - reim[2 * b + 1] * wi;
                float xi = reim[2 * b] * wi + reim[2 * b + 1] * wr;
                reim[2 * b] = reim[2 * a] - xr;
                reim[2 * b + 1] = reim[2 * a + 1] - xi;
                reim[2 * a] += xr;
                reim[2 * a + 1] += xi;
            }
        }
    }
}

void gsv_rfft_amplitude(const gsv_fft_plan *plan, const float *x, float *spec) {
    int n = plan->n;
    float *reim = (float *)malloc(sizeof(float) * 2 * (size_t)n);
    if (!reim) return;
    for (int i = 0; i < n; i++) { reim[2 * i] = x[i]; reim[2 * i + 1] = 0.f; }
    gsv_fft_execute(plan, reim, +1);
    for (int k = 0; k <= n / 2; k++) {
        float r = reim[2 * k], im = reim[2 * k + 1];
        spec[k] = sqrtf(r * r + im * im);
    }
    free(reim);
}
