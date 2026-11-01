/* gru_predict.c - C port of g2p_en.G2p.predict() (float32 GRU enc-dec). */
#include "gru_predict.h"
#include "en_util.h"
#include "en_phonemes.h"
#include <math.h>

#define EMB 256
#define GATE 768      /* 3*EMB */
#define NPH 74        /* decoder phoneme count (fc rows) */

static int g2idx(char c) {
    if (c >= 'a' && c <= 'z') return 3 + (c - 'a');
    return 1;   /* <unk> */
}

/* one GRU cell: h_new = (1-z)*n + z*h ; rzn = x@W.T + b. Float32, sequential dot. */
static void grucell(const float *x, const float *h,
                    const float *w_ih, const float *w_hh,
                    const float *b_ih, const float *b_hh,
                    float *out) {
    float rzn_ih[GATE], rzn_hh[GATE];
    for (int j = 0; j < GATE; j++) {
        float a = 0.f, b = 0.f;
        const float *ri = w_ih + (size_t)j * EMB;
        const float *rh = w_hh + (size_t)j * EMB;
        for (int k = 0; k < EMB; k++) { a += x[k] * ri[k]; b += h[k] * rh[k]; }
        rzn_ih[j] = a + b_ih[j];
        rzn_hh[j] = b + b_hh[j];
    }
    float rz[512];
    for (int j = 0; j < 512; j++) {
        float s = rzn_ih[j] + rzn_hh[j];
        rz[j] = 1.0f / (1.0f + expf(-s));
    }
    /* r = rz[0:256], z = rz[256:512]; n part = rzn[512:768] */
    for (int k = 0; k < EMB; k++) {
        float r = rz[k];
        float z = rz[EMB + k];
        float n = tanhf(rzn_ih[512 + k] + r * rzn_hh[512 + k]);
        out[k] = (1.0f - z) * n + z * h[k];
    }
}

int gsv_gru_predict(const gsv_gru *m, const char **phonemes, const char *word, int *n) {
    int L = 0; while (word[L]) L++;
    float x[EMB], h[EMB];
    for (int k = 0; k < EMB; k++) h[k] = 0.f;

    /* encoder over word chars + </s> (idx 2) */
    for (int t = 0; t <= L; t++) {
        int idx = (t < L) ? g2idx(word[t]) : 2;
        const float *emb = m->enc_emb + (size_t)idx * EMB;
        memcpy(x, emb, sizeof x);
        grucell(x, h, m->enc_w_ih, m->enc_w_hh, m->enc_b_ih, m->enc_b_hh, h);
    }

    /* decoder */
    float dec[EMB];
    memcpy(dec, m->dec_emb + 2 * EMB, sizeof dec);   /* <s> = idx 2 */
    int np = 0;
    for (int i = 0; i < 20; i++) {
        grucell(dec, h, m->dec_w_ih, m->dec_w_hh, m->dec_b_ih, m->dec_b_hh, h);
        float logits[NPH];
        int best = 0;
        for (int j = 0; j < NPH; j++) {
            const float *w = m->fc_w + (size_t)j * EMB;
            float s = m->fc_b[j];
            for (int k = 0; k < EMB; k++) s += h[k] * w[k];
            logits[j] = s;
            if (s > logits[best]) best = j;
        }
        if (best == 3) break;   /* </s> */
        phonemes[np++] = gsv_en_phonemes[best];
        memcpy(dec, m->dec_emb + (size_t)best * EMB, sizeof dec);
    }
    *n = np;
    return 0;
}

/* ---- load ---- */
int gsv_gru_load(gsv_gru *m, const char *dir) {
    memset(m, 0, sizeof *m);
    size_t len = 0;
    char *buf = gsv_en_read_file(dir, "gru.bin", &len);
    if (!buf || len < 8 + 12 * 4) { free(buf); return -1; }
    if (rd_u32(buf) != 0x31555247) { free(buf); return -2; }
    const char *p = buf + 4;
    for (int i = 0; i < 12; i++) p += 4;   /* sizes (validated by layout below) */

    float **blk[12] = { &m->enc_emb, &m->dec_emb, &m->enc_w_ih, &m->enc_w_hh,
                        &m->enc_b_ih, &m->enc_b_hh, &m->dec_w_ih, &m->dec_w_hh,
                        &m->dec_b_ih, &m->dec_b_hh, &m->fc_w, &m->fc_b };
    size_t need[12] = { 29 * EMB, NPH * EMB, GATE * EMB, GATE * EMB, GATE, GATE,
                        GATE * EMB, GATE * EMB, GATE, GATE, NPH * EMB, NPH };
    for (int i = 0; i < 12; i++) {
        if ((size_t)(p - buf) + need[i] * 4 > len) return -3;
        float *a = (float *)malloc(need[i] * 4);
        if (!a) return -4;
        memcpy(a, p, need[i] * 4);
        *blk[i] = a;
        p += need[i] * 4;
    }
    free(buf);
    return 0;
}

void gsv_gru_free(gsv_gru *m) {
    float **blk[12] = { &m->enc_emb, &m->dec_emb, &m->enc_w_ih, &m->enc_w_hh,
                        &m->enc_b_ih, &m->enc_b_hh, &m->dec_w_ih, &m->dec_w_hh,
                        &m->dec_b_ih, &m->dec_b_hh, &m->fc_w, &m->fc_b };
    for (int i = 0; i < 12; i++) { free(*blk[i]); *blk[i] = NULL; }
}
