/* sovits_dec.h - SoVITS decode chain (M5): semantic codes -> audio.
 *
 * Faithful port of SynthesizerTrn.decode for the v2ProPlus batch path:
 *   codebook lookup -> nearest x2 upsample (25Hz->50Hz)
 *   -> enc_p (merged encoder_ssl/text/MRTE/encoder2 + proj, sliced to valid_len by the
 *      graph) -> m_p/logs_p/y_mask @ L0
 *   -> speed interpolation of m/logs (linear; torch F.interpolate, align_corners=False).
 *      Equivalent to interping the pre-proj encoder output: proj is a 1x1 conv and split is
 *      per-channel, both of which commute with linear interpolation.
 *   -> z_p = m + randn * exp(logs) * noise_scale
 *   -> flowdec (flow + HiFi-GAN) -> audio.
 *
 * enc_p/flowdec are dtype-aware: fp32 on CPU, pure-fp16 on CUDA. C-side DSP (interp/noise/
 * sampling) stays fp32; only the tensors crossing into a model are converted to its dtype
 * (see gsv_in_float / gsv_out_f32).
 */
#ifndef GSV_SOVITS_DEC_H
#define GSV_SOVITS_DEC_H

#include "gsvtts.h"

struct gsv_engine;

/* Decode t2s semantic codes to float32 mono audio at the model sample rate.
 * ref_name: key of a warmed-up ref (provides ge/ge512).
 * codes: n_codes token ids from gsv_t2s_generate (2*n_codes <= ly_max).
 * phones2/lt: target-text phoneme ids (G2P output, prompt part excluded).
 * noise_scale: 0 = deterministic (z_p = m_p). speed: 1.0 = normal.
 * out_audio must hold at least n_out samples; *out_len set on success. */
GSVTTS_API int gsv_sovits_decode(struct gsv_engine *e, const char *ref_name,
                                 const int *codes, int n_codes,
                                 const int64_t *phones2, int lt,
                                 float noise_scale, float speed,
                                 float *out_audio, int out_capacity,
                                 int *out_len);

#endif
