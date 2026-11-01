/* features.c - ref warmup feature extraction (ge + prompt codes) */
#include "features.h"
#include "dsp/stft.h"
#include "dsp/kaldi_fbank.h"
#include <stdlib.h>
#include <string.h>

/* ---------------- ge path ---------------- */
int gsv_ref_compute_ge(gsv_engine *e, const gsv_voice_cfg *cfg,
                       const float *wav32k, int64_t n32,
                       const float *wav16k, int64_t n16,
                       gsv_ref *ref) {
    const OrtApi *api = e->api;

    /* 1. STFT amplitude -> mel704 [704, Lm] */
    int n_fft = cfg->spec_filter_length;
    int win_len = cfg->spec_win_length;
    int hop = cfg->spec_hop_length;
    if (n_fft <= 0 || win_len <= 0 || hop <= 0) return GSV_ERR_GENERIC;

    gsv_stft_cfg st;
    st.n_fft = n_fft; st.win_length = win_len; st.hop_length = hop;
    st.window = (float *)malloc(sizeof(float) * (size_t)win_len);
    if (!st.window) return GSV_ERR_NO_MEMORY;
    gsv_stft_make_hann(st.window, win_len);

    int64_t n_frames = gsv_stft_n_frames(&st, n32);
    if (n_frames <= 0) { free(st.window); return GSV_ERR_GENERIC; }
    int n_freq = n_fft / 2 + 1;
    float *spec = (float *)malloc(sizeof(float) * (size_t)n_frames * (size_t)n_freq);
    if (!spec) { free(st.window); return GSV_ERR_NO_MEMORY; }
    if (gsv_stft_amplitude(&st, wav32k, n32, spec) != n_frames) {
        free(spec); free(st.window); return GSV_ERR_GENERIC;
    }

    int mel_bins = 704;
    float *mel704 = (float *)malloc(sizeof(float) * (size_t)mel_bins * (size_t)n_frames);
    if (!mel704) { free(spec); free(st.window); return GSV_ERR_NO_MEMORY; }
    for (int64_t f = 0; f < n_frames; f++)
        for (int c = 0; c < mel_bins; c++)
            mel704[(size_t)c * (size_t)n_frames + (size_t)f] = spec[(size_t)f * (size_t)n_freq + (size_t)c];
    free(spec); free(st.window);

    /* 2. Kaldi fbank80 on 16k -> [80, T] */
    gsv_kaldi_fbank_cfg fb;
    if (gsv_kaldi_fbank_init(&fb, 80, 16000) != 0) { free(mel704); return GSV_ERR_GENERIC; }
    int64_t n_fb = (n16 >= fb.window_size) ? 1 + (n16 - fb.window_size) / fb.shift : 0;
    if (n_fb <= 0) { gsv_kaldi_fbank_destroy(&fb); free(mel704); return GSV_ERR_GENERIC; }
    float *fbank_t = (float *)malloc(sizeof(float) * (size_t)n_fb * 80);
    if (!fbank_t) { gsv_kaldi_fbank_destroy(&fb); free(mel704); return GSV_ERR_NO_MEMORY; }
    if (gsv_kaldi_fbank_compute(&fb, wav16k, n16, fbank_t) != n_fb) {
        free(fbank_t); gsv_kaldi_fbank_destroy(&fb); free(mel704); return GSV_ERR_GENERIC;
    }
    gsv_kaldi_fbank_destroy(&fb);

    /* 3. sv.onnx (pure-fp16): fbank [1,T,80]（时间主序，Kaldi 方向）-> embedding [20480] */
    int emb_dim = e->mf.sv.dim ? e->mf.sv.dim : 20480;
    int64_t fb_dims[3] = {1, n_fb, 80};
    uint16_t *hfb = NULL;
    OrtValue *in_fb = gsv_in_float(api, &e->sv, fbank_t, fb_dims, 3, (size_t)n_fb * 80, &hfb);
    if (!in_fb) { free(hfb); free(fbank_t); free(mel704); return GSV_ERR_BACKEND; }
    const void *fb_in[1] = { in_fb };
    gsv_out sv_out[1];
    int rc = gsv_sess_run(api, &e->sv, fb_in, sv_out, 1);
    gsv_tensor_destroy(api, in_fb);
    free(hfb);
    free(fbank_t);
    if (rc != 0) { free(mel704); return GSV_ERR_BACKEND; }
    float *sv_emb = (float *)malloc(sizeof(float) * (size_t)emb_dim);
    if (!sv_emb) { free(mel704); gsv_sess_free_outputs(api, sv_out, 1); return GSV_ERR_NO_MEMORY; }
    gsv_out_f32(&e->sv, &sv_out[0], sv_emb, (size_t)emb_dim);   /* fp16 -> fp32 */

    /* 4. ge.onnx (pure-fp16): mel704 [1,704,Lm] + sv_emb [20480] -> ge [1,gin,1], ge512 [1,512,1] */
    int64_t mel_dims[3] = {1, 704, n_frames};
    uint16_t *hmel = NULL, *hemb = NULL;
    OrtValue *in_mel = gsv_in_float(api, &e->ge, mel704, mel_dims, 3, (size_t)704 * n_frames, &hmel);
    int64_t emb_dims[1] = { emb_dim };
    OrtValue *in_emb = gsv_in_float(api, &e->ge, sv_emb, emb_dims, 1, (size_t)emb_dim, &hemb);
    if (!in_mel || !in_emb) {
        if (in_mel) gsv_tensor_destroy(api, in_mel);
        if (in_emb) gsv_tensor_destroy(api, in_emb);
        free(hmel); free(hemb);
        free(mel704); free(sv_emb);
        gsv_sess_free_outputs(api, sv_out, 1);
        return GSV_ERR_BACKEND;
    }
    const void *ge_in[2] = { in_mel, in_emb };
    gsv_out ge_out[2];
    rc = gsv_sess_run(api, &e->ge, ge_in, ge_out, 2);
    gsv_tensor_destroy(api, in_mel); free(hmel);
    free(mel704);
    gsv_tensor_destroy(api, in_emb); free(hemb);
    free(sv_emb);
    gsv_sess_free_outputs(api, sv_out, 1);
    if (rc != 0) return GSV_ERR_BACKEND;

    int gin = e->mf.ge.gin_channels ? e->mf.ge.gin_channels : cfg->gin_channels;
    free(ref->ge); free(ref->ge512);
    ref->ge = (float *)malloc(sizeof(float) * (size_t)gin);
    ref->ge512 = (float *)malloc(sizeof(float) * 512);
    if (!ref->ge || !ref->ge512) { gsv_sess_free_outputs(api, ge_out, 2); return GSV_ERR_NO_MEMORY; }
    gsv_out_f32(&e->ge, &ge_out[0], ref->ge, (size_t)gin);        /* fp16 -> fp32 */
    gsv_out_f32(&e->ge, &ge_out[1], ref->ge512, 512);
    ref->ge_dim = gin;
    gsv_sess_free_outputs(api, ge_out, 2);
    return GSV_OK;
}

/* ---------------- prompt codes path ---------------- */
int gsv_ref_compute_codes(gsv_engine *e, const float *wav16k, int64_t n16,
                          gsv_ref *ref) {
    const OrtApi *api = e->api;
    const gsv_vq_cfg *vq = &e->mf.vq;

    /* 1. hubert.onnx (pure-fp16): wav16k [1,S] -> ssl [1,T',768] (fp16 out -> fp32) */
    int dim = vq->dim;
    int64_t w_dims[2] = {1, n16};
    uint16_t *hw = NULL;
    OrtValue *in_w = gsv_in_float(api, &e->hubert, wav16k, w_dims, 2, (size_t)n16, &hw);
    if (!in_w) { free(hw); return GSV_ERR_BACKEND; }
    const void *w_in[1] = { in_w };
    gsv_out ssl_out[1];
    int rc = gsv_sess_run(api, &e->hubert, w_in, ssl_out, 1);
    gsv_tensor_destroy(api, in_w); free(hw);
    if (rc != 0) return GSV_ERR_BACKEND;

    /* ssl layout [1, T', 768] */
    int64_t t_len = ssl_out[0].dims[1];
    float *ssl32 = (float *)malloc(sizeof(float) * (size_t)t_len * (size_t)dim);
    if (!ssl32) { gsv_sess_free_outputs(api, ssl_out, 1); return GSV_ERR_NO_MEMORY; }
    gsv_out_f32(&e->hubert, &ssl_out[0], ssl32, (size_t)t_len * dim);   /* fp16 -> fp32 */
    const float *ssl = ssl32;   /* [T'][768] */

    /* 2. ssl_proj Conv1d(768,768,k=2,s=2) in C: out[t][c] = b[c] + sum_i W[c,i,0]*x[2t+i... ] */
    int k = vq->ssl_proj_kernel ? vq->ssl_proj_kernel : 2;
    int s = vq->ssl_proj_stride ? vq->ssl_proj_stride : 2;
    int64_t out_len = (t_len - k) / s + 1;
    if (out_len <= 0) { free(ssl32); gsv_sess_free_outputs(api, ssl_out, 1); return GSV_ERR_GENERIC; }

    float *proj = (float *)malloc(sizeof(float) * (size_t)out_len * (size_t)dim);
    if (!proj) { free(ssl32); gsv_sess_free_outputs(api, ssl_out, 1); return GSV_ERR_NO_MEMORY; }
    /* Conv1d: out[c,t] = b[c] + sum_in W[c,in,ki] * x[in, t*s+ki];
     * weight layout in ssl_proj.bin: [out_ch, in_ch, kernel]. */
    for (int64_t t = 0; t < out_len; t++) {
        const float *x0 = ssl + (size_t)(t * s) * (size_t)dim;
        const float *x1 = x0 + (size_t)dim;
        for (int c = 0; c < dim; c++) {
            double acc = (double)e->ssl_proj_b[c];
            for (int in_ch = 0; in_ch < dim; in_ch++) {
                const float *w = e->ssl_proj_w + ((size_t)(c * dim + in_ch) * (size_t)k);
                for (int ki = 0; ki < k; ki++)
                    acc += (double)w[ki] * ((ki == 0) ? x0[in_ch] : x1[in_ch]);
            }
            proj[(size_t)t * (size_t)dim + (size_t)c] = (float)acc;
        }
    }
    gsv_sess_free_outputs(api, ssl_out, 1);
    free(ssl32);

    /* 3. VQ encode: nearest codebook entry per frame */
    int bins = vq->bins;
    free(ref->prompt_codes);
    ref->prompt_codes = (int *)malloc(sizeof(int) * (size_t)out_len);
    if (!ref->prompt_codes) { free(proj); return GSV_ERR_NO_MEMORY; }
    for (int64_t t = 0; t < out_len; t++) {
        const float *x = proj + (size_t)t * (size_t)dim;
        int best = 0; float best_d = 1e30f;
        for (int b = 0; b < bins; b++) {
            const float *cb = e->codebook + (size_t)b * (size_t)dim;
            float d = 0.f;
            for (int c = 0; c < dim; c++) { float df = x[c] - cb[c]; d += df * df; }
            if (d < best_d) { best_d = d; best = b; }
        }
        ref->prompt_codes[t] = best;
    }
    ref->n_prompt_codes = (int)out_len;
    free(proj);
    return GSV_OK;
}
