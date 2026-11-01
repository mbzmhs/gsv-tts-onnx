/* gpt.h - t2s GPT autoregressive generation (internal, M4) */
#ifndef GSV_GPT_H
#define GSV_GPT_H

#include "engine.h"

/* Run the text->semantic GPT: prefill over [text phonemes + prompt codes],
 * then a step loop with C-side sampling (top_k/top_p/temperature/repetition).
 *
 * x_ids      : [lx] phoneme ids (int64), values in [0, phoneme_vocab_size)
 * y_ids      : [ly] prompt semantic codes (int64), values in [0, vocab_size)
 * bert       : [lx*1024] float32 per-phoneme BERT features, or NULL for zeros
 * out_tokens : buffer for generated semantic codes (int32), capacity out_capacity
 * out_len    : set to the number of generated codes on success
 *
 * Sampling params come from e->sampling. Returns GSV_OK on success.
 * Exported (also used by the test host for M4 verification). */
GSVTTS_API int gsv_t2s_generate(gsv_engine *e,
                                const int64_t *x_ids, int lx,
                                const int64_t *y_ids, int ly,
                                const float *bert,
                                int *out_tokens, int out_capacity, int *out_len);

#endif
