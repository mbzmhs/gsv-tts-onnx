/* gru_predict.h - C port of g2p_en.G2p.predict(): GRU encoder-decoder over the
 * checkpoint20.npz model (float32, exact operation order). */
#ifndef GSV_G2P_GRU_PREDICT_H
#define GSV_G2P_GRU_PREDICT_H

#include <stdint.h>

typedef struct {
    float *enc_emb;   /* 29 x 256 */
    float *dec_emb;   /* 74 x 256 */
    float *enc_w_ih, *enc_w_hh, *enc_b_ih, *enc_b_hh;  /* 768x256, 768x256, 768, 768 */
    float *dec_w_ih, *dec_w_hh, *dec_b_ih, *dec_b_hh;
    float *fc_w;      /* 74 x 256 */
    float *fc_b;      /* 74 */
} gsv_gru;

int  gsv_gru_load(gsv_gru *m, const char *dir);
void gsv_gru_free(gsv_gru *m);

/* predict(word): returns phoneme strings (from the 74-entry phoneme table) in
 * out (array of >= 20 pointers), sets *n. Empty result if model says </s> at once. */
int gsv_gru_predict(const gsv_gru *m, const char **phonemes, const char *word, int *n);

#endif
