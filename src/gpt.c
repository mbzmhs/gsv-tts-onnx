/* gpt.c - t2s GPT autoregressive generation + C-side token sampling (M4).
 *
 * Faithful port of the reference server's ONNX backend loop
 * (gsv_tts/onnx_backend.py::onnx_infer) with the sampling math from
 * GPT_SoVITS/GPT/utils.py (logits_to_probs + multinomial_sample_one_no_sync).
 */
#include "gpt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GSV_NEG_INF (-1e9f)   /* reference uses -1e9 for bias/masks */

/* RNG lives in engine.c (shared with the SoVITS decode noise path). */
#define rng_uniform gsv_rng_uniform

/* ---------------- sampling (port of GPT/utils.py) ---------------- */
typedef struct { float v; int idx; } fv_t;

static int cmp_fv_desc(const void *a, const void *b) {
    float va = ((const fv_t *)a)->v, vb = ((const fv_t *)b)->v;
    if (va < vb) return 1;
    if (va > vb) return -1;
    return 0;
}

/* Deterministic sampling transforms, in the reference's exact order:
 *   repetition_penalty -> top_p (raw logits) -> temperature -> top_k -> softmax.
 * logits is modified in place; probs[vocab] receives the final probabilities.
 * Returns 0 on success, -1 if no finite probability remains. */
static int sample_transforms(gsv_engine *e, float *logits, int vocab,
                             const int64_t *prev, int prev_len,
                             const gsv_sampling *s, float *probs) {
    (void)e;
    /* 1. repetition penalty */
    if (prev_len > 0 && s->repetition_penalty != 1.0f) {
        float rep = s->repetition_penalty;
        for (int i = 0; i < prev_len; i++) {
            int t = (int)prev[i];
            if (t >= 0 && t < vocab) {
                float sc = logits[t];
                logits[t] = (sc < 0.0f) ? sc * rep : sc / rep;
            }
        }
    }

    fv_t *arr = (fv_t *)malloc(sizeof(fv_t) * (size_t)vocab);
    if (!arr) return -1;

    /* 2. top_p (nucleus) on raw logits */
    if (s->top_p < 1.0f) {
        for (int i = 0; i < vocab; i++) { arr[i].v = logits[i]; arr[i].idx = i; }
        qsort(arr, (size_t)vocab, sizeof(fv_t), cmp_fv_desc);
        float m = arr[0].v;
        for (int i = 1; i < vocab; i++) if (arr[i].v > m) m = arr[i].v;
        float sum = 0.0f;
        for (int i = 0; i < vocab; i++) { arr[i].v = expf(arr[i].v - m); sum += arr[i].v; }
        float cum = 0.0f;
        for (int r = 0; r < vocab; r++) {
            float p = arr[r].v / sum;
            if (r > 0 && cum + p > s->top_p) logits[arr[r].idx] = GSV_NEG_INF;
            cum += p;
        }
    }

    /* 3. temperature */
    float temp = s->temperature > 1e-5f ? s->temperature : 1e-5f;
    for (int i = 0; i < vocab; i++) logits[i] /= temp;

    /* 4. top_k */
    if (s->top_k > 0) {
        int k = s->top_k < vocab ? s->top_k : vocab;
        for (int i = 0; i < vocab; i++) { arr[i].v = logits[i]; arr[i].idx = i; }
        qsort(arr, (size_t)vocab, sizeof(fv_t), cmp_fv_desc);
        float pivot = arr[k - 1].v;
        for (int i = 0; i < vocab; i++)
            if (logits[i] < pivot) logits[i] = GSV_NEG_INF;
    }

    free(arr);

    /* 5. softmax */
    float m = -INFINITY;
    for (int i = 0; i < vocab; i++) if (logits[i] > m) m = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (m == -INFINITY || logits[i] <= GSV_NEG_INF) continue;
        sum += expf(logits[i] - m);
    }
    if (sum <= 0.0f) return -1;
    for (int i = 0; i < vocab; i++) {
        probs[i] = (logits[i] <= GSV_NEG_INF) ? 0.0f : expf(logits[i] - m) / sum;
    }
    return 0;
}

/* Sample one token from logits[vocab] (modified in place):
 * deterministic transforms + multinomial via the exponential race
 * (multinomial_sample_one_no_sync: q ~ Exp(1), argmax(probs / q)). */
static int sample_token(gsv_engine *e, float *logits, int vocab,
                        const int64_t *prev, int prev_len,
                        const gsv_sampling *s) {
    float *probs = (float *)malloc(sizeof(float) * (size_t)vocab);
    if (!probs) return 0;
    if (sample_transforms(e, logits, vocab, prev, prev_len, s, probs) != 0) {
        free(probs);
        return 0;
    }
    int best = 0;
    float best_r = -1.0f;
    for (int i = 0; i < vocab; i++) {
        if (probs[i] <= 0.0f) continue;
        float q = -logf(rng_uniform(e));
        float r = probs[i] / q;
        if (r > best_r) { best_r = r; best = i; }
    }
    free(probs);
    return best;
}

/* ---------------- t2s generation ---------------- */
int gsv_t2s_generate(gsv_engine *e, const int64_t *x_ids, int lx,
                     const int64_t *y_ids, int ly, const float *bert,
                     int *out_tokens, int out_capacity, int *out_len) {
    if (!e || !e->has_voice) return GSV_ERR_NOT_LOADED;
    if (!x_ids || !y_ids || !out_tokens || !out_len) return GSV_ERR_INVALID_ARG;
    const gsv_t2s_cfg *t = &e->mf.t2s;
    int max_len = t->max_len, L = t->num_layers, H = t->num_heads, D = t->model_dim;
    int hd = D / H, vocab = t->vocab_size, eos = t->eos;
    const OrtApi *api = e->api;

    int s = lx + ly;
    if (lx <= 0 || ly < 0 || s >= max_len) {
        gsv_set_error(e, "gsv_t2s_generate: prompt length %d out of range [1,%d)",
                      s, max_len);
        return GSV_ERR_INVALID_ARG;
    }

    /* lazily allocated scratch (freed on model unload). t2s is pure-fp16, so the KV cache
     * is stored in fp16 and fed/copied directly (no per-step big-tensor conversion). */
    size_t kb = (size_t)L * H * (size_t)max_len * (size_t)hd;
    if (!e->t2s_kbuf) {
        e->t2s_kbuf = (uint16_t *)malloc(kb * sizeof(uint16_t));
        e->t2s_vbuf = (uint16_t *)malloc(kb * sizeof(uint16_t));
        e->t2s_idx  = (int64_t *)malloc((size_t)max_len * sizeof(int64_t));
        if (!e->t2s_kbuf || !e->t2s_vbuf || !e->t2s_idx) return GSV_ERR_NO_MEMORY;
        for (int i = 0; i < max_len; i++) e->t2s_idx[i] = i;   /* arange */
    }

    /* prompt attention bias [1,1,s,s]: all rows attend to text cols; audio rows
     * causal within themselves. */
    float *bias = (float *)malloc(sizeof(float) * (size_t)s * (size_t)s);
    if (!bias) return GSV_ERR_NO_MEMORY;
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) {
            int ok = (j < lx) || (i >= lx && i >= j);
            bias[(size_t)i * (size_t)s + (size_t)j] = ok ? 0.0f : GSV_NEG_INF;
        }

    float *bert_buf;
    int own_bert = 0;
    if (bert) {
        bert_buf = (float *)bert;
    } else {
        bert_buf = (float *)calloc((size_t)lx * 1024, sizeof(float));
        own_bert = 1;
        if (!bert_buf) { free(bias); return GSV_ERR_NO_MEMORY; }
    }

    /* ---- prefill (t2s pure-fp16: bert/attn_bias -> fp16; logits/KV out are fp16) ---- */
    int64_t xd[2] = {1, lx}, yd[2] = {1, ly};
    int64_t bd[3] = {1, lx, 1024}, ad[4] = {1, 1, s, s};
    uint16_t *hbert = NULL, *hbias = NULL;
    OrtValue *ix = gsv_tensor_create(api, x_ids, xd, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    OrtValue *iy = gsv_tensor_create(api, y_ids, yd, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    OrtValue *ib = gsv_in_float(api, &e->t2s_prefill, bert_buf, bd, 3, (size_t)lx * 1024, &hbert);
    OrtValue *ia = gsv_in_float(api, &e->t2s_prefill, bias, ad, 4, (size_t)s * (size_t)s, &hbias);
    if (!ix || !iy || !ib || !ia) {
        gsv_tensor_destroy(api, ix); gsv_tensor_destroy(api, iy);
        gsv_tensor_destroy(api, ib); gsv_tensor_destroy(api, ia);
        free(hbert); free(hbias);
        if (own_bert) free(bert_buf);
        free(bias);
        return GSV_ERR_BACKEND;
    }
    const void *pin[4] = { ix, iy, ib, ia };   /* contract order: x_ids,y_ids,bert,attn_bias */
    gsv_out po[3];
    int rc = gsv_sess_run(api, &e->t2s_prefill, pin, po, 3);
    gsv_tensor_destroy(api, ix); gsv_tensor_destroy(api, iy);
    gsv_tensor_destroy(api, ib); gsv_tensor_destroy(api, ia);
    free(hbert); free(hbias);
    if (own_bert) free(bert_buf);
    free(bias);
    if (rc != 0) {
        gsv_set_error(e, "gsv_t2s_generate: prefill run failed");
        return GSV_ERR_BACKEND;
    }

    float *cur = (float *)malloc(sizeof(float) * (size_t)vocab);
    if (!cur) { gsv_sess_free_outputs(api, po, 3); return GSV_ERR_NO_MEMORY; }
    gsv_out_f32(&e->t2s_prefill, &po[0], cur, (size_t)vocab);   /* logits fp16 -> fp32 */

    /* present_k/v [L,1,H,s,hd] fp16 -> buffers [L,1,H,max_len,hd] at [:s] (direct copy) */
    for (int lh = 0; lh < L * H; lh++) {
        memcpy(e->t2s_kbuf + ((size_t)lh * (size_t)max_len) * (size_t)hd,
               (const uint16_t *)po[1].data + ((size_t)lh * (size_t)s) * (size_t)hd,
               sizeof(uint16_t) * (size_t)s * (size_t)hd);
        memcpy(e->t2s_vbuf + ((size_t)lh * (size_t)max_len) * (size_t)hd,
               (const uint16_t *)po[2].data + ((size_t)lh * (size_t)s) * (size_t)hd,
               sizeof(uint16_t) * (size_t)s * (size_t)hd);
    }
    gsv_sess_free_outputs(api, po, 3);

    int kv = s;

    /* previous tokens (prompt + generated so far) for repetition penalty */
    int64_t *prev = (int64_t *)malloc(sizeof(int64_t) * (size_t)(max_len + 2));
    if (!prev) { free(cur); return GSV_ERR_NO_MEMORY; }
    memcpy(prev, y_ids, sizeof(int64_t) * (size_t)ly);
    int prev_len = ly;

    /* first sample: reference masks suppressed tokens and samples logits[:, :-1]
     * (eos excluded) */
    for (int si = 0; si < t->n_suppressed; si++)
        cur[t->suppressed[si]] = GSV_NEG_INF;
    cur[eos] = GSV_NEG_INF;
    int64_t tok = sample_token(e, cur, vocab, prev, prev_len, &e->sampling);
    prev[prev_len++] = tok;

    /* ---- step loop ---- */
    int n_iter = 0;
    int max_iter = max_len - s;
    int64_t kdim[5] = {L, 1, H, max_len, hd};
    for (int idx = 1; idx <= max_iter; idx++) {
        int64_t tid[2] = {1, 1};
        int64_t posv = kv - lx;
        int64_t pdim[1] = {1};
        int64_t idim[1] = {kv};
        OrtValue *itok = gsv_tensor_create(api, &tok, tid, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        OrtValue *ipos = gsv_tensor_create(api, &posv, pdim, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        OrtValue *iidx = gsv_tensor_create(api, e->t2s_idx, idim, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        OrtValue *ik   = gsv_tensor_create(api, e->t2s_kbuf, kdim, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        OrtValue *iv   = gsv_tensor_create(api, e->t2s_vbuf, kdim, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        if (!itok || !ipos || !iidx || !ik || !iv) {
            gsv_tensor_destroy(api, itok); gsv_tensor_destroy(api, ipos);
            gsv_tensor_destroy(api, iidx); gsv_tensor_destroy(api, ik);
            gsv_tensor_destroy(api, iv);
            free(prev); free(cur);
            return GSV_ERR_BACKEND;
        }
        const void *sin[5] = { itok, ipos, iidx, ik, iv };   /* token_id,pos_index,idx,past_k,past_v */
        gsv_out so[3];
        rc = gsv_sess_run(api, &e->t2s_step, sin, so, 3);
        gsv_tensor_destroy(api, itok); gsv_tensor_destroy(api, ipos);
        gsv_tensor_destroy(api, iidx); gsv_tensor_destroy(api, ik);
        gsv_tensor_destroy(api, iv);
        if (rc != 0) {
            gsv_set_error(e, "gsv_t2s_generate: step run failed at idx %d", idx);
            free(prev); free(cur);
            return GSV_ERR_BACKEND;
        }

        gsv_out_f32(&e->t2s_step, &so[0], cur, (size_t)vocab);   /* logits fp16 -> fp32 */
        /* k_new/v_new [L,1,H,1,hd] fp16 -> buffers at position kv (direct copy) */
        for (int lh = 0; lh < L * H; lh++) {
            memcpy(e->t2s_kbuf + ((size_t)lh * (size_t)max_len + (size_t)kv) * (size_t)hd,
                   (const uint16_t *)so[1].data + (size_t)lh * (size_t)hd,
                   sizeof(uint16_t) * (size_t)hd);
            memcpy(e->t2s_vbuf + ((size_t)lh * (size_t)max_len + (size_t)kv) * (size_t)hd,
                   (const uint16_t *)so[2].data + (size_t)lh * (size_t)hd,
                   sizeof(uint16_t) * (size_t)hd);
        }
        gsv_sess_free_outputs(api, so, 3);
        kv++;
        n_iter = idx;

        /* reference: initial_suppression_steps=10, check_interval=5 */
        if (idx < 10)
            for (int si = 0; si < t->n_suppressed; si++)
                cur[t->suppressed[si]] = GSV_NEG_INF;
        tok = sample_token(e, cur, vocab, prev, prev_len, &e->sampling);
        prev[prev_len++] = tok;

        /* stop at the FIRST sampled eos, every step. Checking only every 5th step
         * let eos escape (sampled off-checkpoint, not re-sampled at the next one),
         * burning max_len-s steps (~7s GPU) per chunk; the finalize truncates at
         * the first eos anyway, so audio is bit-identical, minus the wasted steps. */
        if (tok == eos) break;
    }

    /* ---- finalize: pre_tokens[:, -n_iter:] truncated at first eos ---- */
    int start = prev_len - n_iter;   /* drops prompt + first generated token */
    int n_keep = n_iter;
    for (int i = 0; i < n_keep; i++)
        if (prev[start + i] == eos) { n_keep = i; break; }
    if (n_keep > out_capacity) {
        gsv_set_error(e, "gsv_t2s_generate: output capacity %d < %d tokens",
                      out_capacity, n_keep);
        free(prev); free(cur);
        return GSV_ERR_NO_MEMORY;
    }
    for (int i = 0; i < n_keep; i++) out_tokens[i] = (int)prev[start + i];
    *out_len = n_keep;
    free(prev);
    free(cur);
    return GSV_OK;
}
