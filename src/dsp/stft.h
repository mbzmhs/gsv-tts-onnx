/* stft.h - amplitude STFT matching torchaudio.transforms.Spectrogram
 * (hann periodic window, center=True reflect padding, power=1) */
#ifndef GSV_STFT_H
#define GSV_STFT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_fft;
    int win_length;
    int hop_length;
    float *window;   /* [win_length] periodic hann */
} gsv_stft_cfg;

/* out: caller-provided float buffer of size (n_frames * (n_fft/2+1)).
 * Returns number of frames, or -1 on error.
 * n_frames = 1 + (padded_len - n_fft) / hop, padded_len = len + 2*(n_fft/2). */
int64_t gsv_stft_amplitude(const gsv_stft_cfg *cfg, const float *x, int64_t len, float *out);

/* Number of frames for a signal of len samples. */
int64_t gsv_stft_n_frames(const gsv_stft_cfg *cfg, int64_t len);

/* Fill w[n] with periodic hann (torch.hann_window default). */
void gsv_stft_make_hann(float *w, int n);

#ifdef __cplusplus
}
#endif

#endif
