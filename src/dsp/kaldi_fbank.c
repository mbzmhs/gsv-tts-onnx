/* kaldi_fbank.c - Kaldi log-fbank (torchaudio.compliance.kaldi.fbank port) */
#include "kaldi_fbank.h"
#include "fft.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FLT_EPS 1.19209290e-07f

static double hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }

int gsv_kaldi_fbank_init(gsv_kaldi_fbank_cfg *cfg, int num_bins, int sample_rate) {
    cfg->num_bins = num_bins;
    int window_size = (int)lround(0.025 * sample_rate);   /* 25ms */
    int shift = (int)lround(0.010 * sample_rate);         /* 10ms */
    int padded = 1;
    while (padded < window_size) padded <<= 1;            /* next pow2 */
    cfg->window_size = window_size;
    cfg->shift = shift;
    cfg->padded_window = padded;

    int num_fft_bins = padded / 2;                        /* freq grid size (excl. nyquist col) */
    double nyquist = 0.5 * sample_rate;
    double low_freq = 20.0, high_freq = nyquist;          /* high_freq<=0 -> += nyquist */
    double fft_bin_width = (double)sample_rate / padded;

    double mel_low = hz2mel(low_freq), mel_high = hz2mel(high_freq);
    double mel_delta = (mel_high - mel_low) / (num_bins + 1);

    cfg->mel_banks = (float *)malloc(sizeof(float) * num_bins * (num_fft_bins + 1));
    if (!cfg->mel_banks) return -1;

    for (int b = 0; b < num_bins; b++) {
        double left_mel = mel_low + b * mel_delta;
        double center_mel = mel_low + (b + 1.0) * mel_delta;
        double right_mel = mel_low + (b + 2.0) * mel_delta;
        float *row = cfg->mel_banks + (size_t)b * (num_fft_bins + 1);
        for (int f = 0; f < num_fft_bins; f++) {
            double mel = hz2mel(fft_bin_width * f);
            double up = (mel - left_mel) / (center_mel - left_mel);
            double down = (right_mel - mel) / (right_mel - center_mel);
            double v = up < down ? up : down;
            if (v < 0.0) v = 0.0;
            row[f] = (float)v;
        }
        row[num_fft_bins] = 0.0f;   /* pad right column with zero */
    }
    return 0;
}

void gsv_kaldi_fbank_destroy(gsv_kaldi_fbank_cfg *cfg) {
    if (cfg->mel_banks) free(cfg->mel_banks);
    cfg->mel_banks = NULL;
}

int64_t gsv_kaldi_fbank_compute(const gsv_kaldi_fbank_cfg *cfg, const float *x16k,
                                int64_t len, float *out) {
    if (len < cfg->window_size) return 0;
    int64_t m = 1 + (len - cfg->window_size) / cfg->shift;   /* snip_edges=True */
    int ws = cfg->window_size, padded = cfg->padded_window, nb = cfg->num_bins;

    static gsv_fft_plan plan;
    static int plan_n = 0;
    if (plan_n != padded) {
        if (plan_n) gsv_fft_plan_destroy(&plan);
        if (gsv_fft_plan_create(&plan, padded) != 0) return -1;
        plan_n = padded;
    }

    /* povey window = hann(periodic=False)^0.85 */
    float *win = (float *)malloc(sizeof(float) * ws);
    for (int i = 0; i < ws; i++) {
        double h = 0.5 * (1.0 - cos(2.0 * M_PI * i / (ws - 1)));
        win[i] = (float)(pow(h, 0.85));
    }

    float *frame = (float *)malloc(sizeof(float) * padded);
    float *spec = (float *)malloc(sizeof(float) * (padded / 2 + 1));
    if (!frame || !spec) { free(win); free(frame); free(spec); return -1; }

    /* Kaldi order per frame: raw -> remove_dc_offset -> pre-emphasis (left edge
     * replicate-padded, computed from the dc-subtracted values) -> window. */
    for (int64_t t = 0; t < m; t++) {
        int64_t start = t * cfg->shift;
        double mean = 0.0;
        for (int i = 0; i < ws; i++) mean += x16k[start + i];
        mean /= ws;   /* remove_dc_offset */
        for (int i = 0; i < ws; i++) frame[i] = (float)(x16k[start + i] - mean);
        float prev = frame[0];   /* replicate pad at left edge */
        for (int i = 0; i < ws; i++) {
            float cur = frame[i];
            frame[i] = cur - 0.97f * prev;
            prev = cur;
        }
        for (int i = 0; i < ws; i++) frame[i] *= win[i];
        for (int i = ws; i < padded; i++) frame[i] = 0.0f;

        /* power spectrum: |rfft|^2 */
        gsv_rfft_amplitude(&plan, frame, spec);
        for (int f = 0; f <= padded / 2; f++) spec[f] = spec[f] * spec[f];

        for (int b = 0; b < nb; b++) {
            const float *row = cfg->mel_banks + (size_t)b * (padded / 2 + 1);
            double acc = 0.0;
            for (int f = 0; f <= padded / 2; f++) acc += (double)spec[f] * row[f];
            if (acc < FLT_EPS) acc = FLT_EPS;
            out[t * nb + b] = (float)log(acc);
        }
    }

    free(win); free(frame); free(spec);
    return m;
}
