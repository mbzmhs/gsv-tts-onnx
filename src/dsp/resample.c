/* resample.c - exact port of torchaudio sinc_interp_hann resampler */
#include "resample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int gcd_int(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a < 0 ? -a : a; }

int gsv_resampler_create(gsv_resampler *r, int sr0, int sr1) {
    if (sr0 <= 0 || sr1 <= 0) return -1;
    r->sr0 = sr0; r->sr1 = sr1;
    if (sr0 == sr1) { r->o = r->n = 1; r->width = 0; r->klen = 1; r->kernels = NULL; return 0; }

    int g = gcd_int(sr0, sr1);
    int o = sr0 / g, n = sr1 / g;
    const double LPW = 6.0;      /* lowpass_filter_width */
    const double ROLLOFF = 0.99;
    r->o = o; r->n = n;

    double base_freq = (double)(o < n ? o : n) * ROLLOFF;
    int width = (int)ceil(LPW * (double)o / base_freq);
    int klen = 2 * width + o;
    r->width = width; r->klen = klen;

    r->kernels = (float *)malloc(sizeof(float) * (size_t)n * klen);
    if (!r->kernels) return -1;

    /* idx[k] = (k - width) / o,  k in [0, klen)   (arange(-width, width+o)/orig_freq) */
    double scale = base_freq / o;
    for (int j = 0; j < n; j++) {
        float *row = r->kernels + (size_t)j * klen;
        for (int k = 0; k < klen; k++) {
            double idx = ((double)k - width) / o;
            double t = (-j) / (double)n + idx;   /* arange(0,-n,-1)/n + idx */
            t *= base_freq;
            if (t > LPW) t = LPW;
            if (t < -LPW) t = -LPW;
            double window = cos(t * M_PI / LPW / 2.0);
            window = window * window;             /* hann: cos^2 */
            double tr = t * M_PI;
            double sinc = (tr == 0.0) ? 1.0 : sin(tr) / tr;
            row[k] = (float)(sinc * window * scale);
        }
    }
    return 0;
}

void gsv_resampler_destroy(gsv_resampler *r) {
    if (r->kernels) free(r->kernels);
    r->kernels = NULL;
}

int64_t gsv_resampler_out_len(const gsv_resampler *r, int64_t n) {
    if (r->o == 1 && r->n == 1) return n;
    return (n * r->n + r->o - 1) / r->o;  /* ceil(n*n/o) */
}

float *gsv_resample(const gsv_resampler *r, const float *x, int64_t n, int64_t *out_n) {
    if (r->o == 1 && r->n == 1) {
        float *y = (float *)malloc(sizeof(float) * (size_t)n);
        memcpy(y, x, sizeof(float) * (size_t)n);
        *out_n = n;
        return y;
    }
    int64_t target = gsv_resampler_out_len(r, n);
    float *y = (float *)calloc((size_t)target, sizeof(float));
    if (!y) { *out_n = 0; return NULL; }

    for (int64_t j = 0; j < target; j++) {
        const float *row = r->kernels + (size_t)(j % r->n) * r->klen;
        int64_t base = (j / r->n) * r->o - r->width;   /* x index of kernel[0] (q*o, q=j/n) */
        double acc = 0.0;
        for (int k = 0; k < r->klen; k++) {
            int64_t i = base + k;
            if (i >= 0 && i < n) acc += (double)x[i] * row[k];
        }
        y[j] = (float)acc;
    }
    *out_n = target;
    return y;
}
