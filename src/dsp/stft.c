/* stft.c - amplitude STFT (torchaudio Spectrogram semantics) */
#include "stft.h"
#include "fft.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* numpy/torch 'reflect' index mapping (no edge repeat) */
static int reflect_idx(int p, int len) {
    if (len <= 1) return 0;
    int period = 2 * (len - 1);
    p %= period;
    if (p < 0) p += period;
    if (p < len) return p;
    return period - p;
}

int64_t gsv_stft_n_frames(const gsv_stft_cfg *cfg, int64_t len) {
    int64_t pad = cfg->n_fft / 2;
    int64_t padded_len = len + 2 * pad;
    if (padded_len < cfg->n_fft) return 0;
    return 1 + (padded_len - cfg->n_fft) / cfg->hop_length;
}

int64_t gsv_stft_amplitude(const gsv_stft_cfg *cfg, const float *x, int64_t len, float *out) {
    if (len <= 0) return 0;
    static gsv_fft_plan plan;
    static int plan_n = 0;
    if (plan_n != cfg->n_fft) {
        if (plan_n) gsv_fft_plan_destroy(&plan);
        if (gsv_fft_plan_create(&plan, cfg->n_fft) != 0) return -1;
        plan_n = cfg->n_fft;
    }

    int64_t n_frames = gsv_stft_n_frames(cfg, len);
    if (n_frames <= 0) return 0;
    int half = cfg->n_fft / 2 + 1;

    float *frame = (float *)malloc(sizeof(float) * cfg->n_fft);
    if (!frame) return -1;

    for (int64_t t = 0; t < n_frames; t++) {
        int64_t start = t * cfg->hop_length; /* index in padded signal */
        for (int i = 0; i < cfg->n_fft; i++) {
            int idx = reflect_idx((int)(start + i - cfg->n_fft / 2), (int)len);
            float v = x[idx];
            if (i < cfg->win_length) v *= cfg->window[i];
            frame[i] = v;
        }
        gsv_rfft_amplitude(&plan, frame, out + t * half);
    }
    free(frame);
    return n_frames;
}

/* helper: build a periodic hann window of given length */
void gsv_stft_make_hann(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
}
