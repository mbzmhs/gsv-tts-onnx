/* kaldi_fbank.h - Kaldi-compatible log-fbank (torchaudio.compliance.kaldi.fbank)
 * fixed params of the SV path: sr=16000, num_mel_bins=80, dither=0,
 * frame_length=25ms, frame_shift=10ms, preemphasis=0.97, povey window,
 * snip_edges=True, use_power=True, use_log_fbank=True */
#ifndef GSV_KALDI_FBANK_H
#define GSV_KALDI_FBANK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_bins;        /* 80 */
    int window_size;     /* 400 @16k */
    int padded_window;   /* 512 (next pow2) */
    int shift;           /* 160 @16k */
    float *mel_banks;    /* [num_bins][padded/2+1] (last col zero) */
} gsv_kaldi_fbank_cfg;

int gsv_kaldi_fbank_init(gsv_kaldi_fbank_cfg *cfg, int num_bins, int sample_rate);
void gsv_kaldi_fbank_destroy(gsv_kaldi_fbank_cfg *cfg);

/* out: caller buffer of size n_frames*num_bins. Returns n_frames (0 if too short). */
int64_t gsv_kaldi_fbank_compute(const gsv_kaldi_fbank_cfg *cfg, const float *x16k,
                                int64_t len, float *out);

#ifdef __cplusplus
}
#endif

#endif
