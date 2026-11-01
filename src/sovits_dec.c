/* sovits_dec.c - SoVITS decode chain (M5): semantic codes -> audio. */
#include "engine.h"
#include "sovits_dec.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* torch F.interpolate(mode="linear", align_corners=False) on the last dim.
 * w = (L/L1)*(j+0.5) - 0.5; boundary weights clamp to the edge value. */
static void interp_linear_f32(const float *x, int L, float *y, int L1) {
    float scale = (float)L / (float)L1;
    for (int j = 0; j < L1; j++) {
        float w = scale * (j + 0.5f) - 0.5f;
        if (w <= 0.0f) { y[j] = x[0]; continue; }
        if (w >= (float)(L - 1)) { y[j] = x[L - 1]; continue; }
        int i0 = (int)w;
        float f = w - (float)i0;
        y[j] = x[i0] * (1.0f - f) + x[i0 + 1] * f;
    }
}

/* torch F.interpolate(mode="nearest", align_corners=False) on the last dim. */
static void interp_nearest_f32(const float *x, int L, float *y, int L1) {
    float scale = (float)L / (float)L1;
    for (int j = 0; j < L1; j++) {
        int i = (int)(scale * (j + 0.5f));
        if (i < 0) i = 0;
        if (i >= L) i = L - 1;
        y[j] = x[i];
    }
}

static const gsv_ref *find_ref(gsv_engine *e, const char *ref_name) {
    for (int i = 0; i < e->voice.n_refs; i++)
        if (strcmp(e->voice.refs[i].name, ref_name) == 0)
            return &e->voice.refs[i];
    return NULL;
}

int gsv_sovits_decode(gsv_engine *e, const char *ref_name,
                      const int *codes, int n_codes,
                      const int64_t *phones2, int lt,
                      float noise_scale, float speed,
                      float *out_audio, int out_capacity, int *out_len) {
    if (!e || !e->has_voice) return GSV_ERR_NOT_LOADED;
    if (!ref_name || !codes || !phones2 || !out_audio || !out_len)
        return GSV_ERR_INVALID_ARG;
    const gsv_ref *ref = find_ref(e, ref_name);
    if (!ref || !ref->ge || !ref->ge512) return GSV_ERR_REF_MISSING;

    const gsv_voice_cfg *cfg = &e->voice.cfg;
    const OrtApi *api = e->api;
    int dim = e->mf.vq.dim;              /* 768 */
    int bins = e->mf.vq.bins;            /* 1024 */
    int inter = cfg->inter_channels;     /* 192 */
    int ly_max = cfg->ly_max, lt_max = cfg->lt_max;
    int gin = ref->ge_dim;

    if (n_codes <= 0 || lt < 1 || speed <= 0.0f) return GSV_ERR_INVALID_ARG;
    int L0 = 2 * n_codes;                /* 25Hz -> 50Hz nearest x2 */
    if (L0 > ly_max) {
        gsv_set_error(e, "gsv_sovits_decode: %d codes need %d frames > ly_max %d",
                      n_codes, L0, ly_max);
        return GSV_ERR_INVALID_ARG;
    }
    if (lt > lt_max) {
        gsv_set_error(e, "gsv_sovits_decode: %d phones > lt_max %d", lt, lt_max);
        return GSV_ERR_INVALID_ARG;
    }
    for (int i = 0; i < n_codes; i++)
        if (codes[i] < 0 || codes[i] >= bins) {
            gsv_set_error(e, "gsv_sovits_decode: code %d out of [0,%d)", codes[i], bins);
            return GSV_ERR_INVALID_ARG;
        }

    /* ---- scratch ---- */
    float *qp = (float *)calloc((size_t)dim * ly_max, sizeof(float));   /* [768][ly_max] */
    float *ymask_p = (float *)malloc(sizeof(float) * (size_t)ly_max);
    int64_t *txt_ids = (int64_t *)malloc(sizeof(int64_t) * (size_t)lt_max);
    float *tmask = (float *)malloc(sizeof(float) * (size_t)lt_max);
    if (!qp || !ymask_p || !txt_ids || !tmask) { free(qp); free(ymask_p); free(txt_ids); free(tmask); return GSV_ERR_NO_MEMORY; }

    /* codebook lookup + nearest x2 upsample: frames 2i, 2i+1 = codebook[codes[i]] */
    for (int i = 0; i < n_codes; i++) {
        const float *v = e->codebook + (size_t)codes[i] * (size_t)dim;
        for (int c = 0; c < dim; c++) {
            qp[(size_t)c * ly_max + 2 * i] = v[c];
            qp[(size_t)c * ly_max + 2 * i + 1] = v[c];
        }
    }
    for (int t = 0; t < ly_max; t++) ymask_p[t] = (t < L0) ? 1.0f : 0.0f;
    for (int i = 0; i < lt_max; i++) {
        txt_ids[i] = (i < lt) ? phones2[i] : 0;
        tmask[i] = (i < lt) ? 1.0f : 0.0f;
    }
    int64_t valid_len = L0;

    /* enc_p (merged encoder+proj) outputs m/logs at L0; speed interp then runs C-side on
     * m/logs (equivalent to interping the pre-proj encoder output: proj is a 1x1 conv and
     * split is per-channel, both commute with linear interpolation). The flowdec mask is
     * all-ones over [0,L1) — the same value the old encp_b returned for an all-ones input. */
    float *m0 = NULL, *logs0 = NULL;     /* [inter*L0] fp32 from enc_p */
    float *m1 = NULL, *logs1 = NULL;     /* [inter*L1] fp32 (== m0/logs0 when speed==1) */
    float *fmask1 = NULL;                /* [L1] ones for flowdec */
    float *zp = NULL;                    /* [inter*L1] */
    int L1 = 0;
    gsv_out eo[3] = {0}, fo[1] = {0};
    uint16_t *hq=NULL, *hge=NULL, *hy=NULL, *htm=NULL;   /* enc_p fp16 input scratch */
    uint16_t *hz=NULL, *hfm=NULL, *hge2=NULL;            /* flowdec fp16 input scratch */
    OrtValue *iq=NULL,*it=NULL,*ig=NULL,*iy=NULL,*im=NULL,*iv=NULL;
    OrtValue *iz=NULL,*ifm=NULL,*ifg=NULL;
    int rc = GSV_ERR_BACKEND;

    /* ---- enc_p: (quantized,text_ids,ge512,y_mask,t_mask,valid_len) -> m,logs,mask @ L0 ---- */
    int64_t qd[3] = {1, dim, ly_max};
    int64_t td[2] = {1, lt_max};
    int64_t gd[3] = {1, 512, 1};
    int64_t ymd[3] = {1, 1, ly_max};
    int64_t tmd[3] = {1, 1, lt_max};
    int64_t vd[1] = {1};
    iq = gsv_in_float(api, &e->voice.enc_p, qp, qd, 3, (size_t)dim * ly_max, &hq);
    it = gsv_tensor_create(api, txt_ids, td, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ig = gsv_in_float(api, &e->voice.enc_p, ref->ge512, gd, 3, 512, &hge);
    iy = gsv_in_float(api, &e->voice.enc_p, ymask_p, ymd, 3, (size_t)ly_max, &hy);
    im = gsv_in_float(api, &e->voice.enc_p, tmask, tmd, 3, (size_t)lt_max, &htm);
    iv = gsv_tensor_create(api, &valid_len, vd, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!iq || !it || !ig || !iy || !im || !iv) goto cleanup;
    const void *ein[6] = { iq, it, ig, iy, im, iv };
    rc = gsv_sess_run(api, &e->voice.enc_p, ein, eo, 3);
    gsv_tensor_destroy(api, iq); gsv_tensor_destroy(api, it); gsv_tensor_destroy(api, ig);
    gsv_tensor_destroy(api, iy); gsv_tensor_destroy(api, im); gsv_tensor_destroy(api, iv);
    free(hq); free(hge); free(hy); free(htm); hq=hge=hy=htm=NULL; iq=it=ig=iy=im=iv=NULL;
    if (rc != 0) { gsv_set_error(e, "gsv_sovits_decode: enc_p run failed"); goto cleanup; }

    int Ly = (int)eo[0].dims[2];
    if (Ly != L0) {
        gsv_set_error(e, "gsv_sovits_decode: enc_p output len %d != %d", Ly, L0);
        goto cleanup;
    }
    m0 = (float *)malloc(sizeof(float) * (size_t)inter * (size_t)L0);
    logs0 = (float *)malloc(sizeof(float) * (size_t)inter * (size_t)L0);
    if (!m0 || !logs0) { rc = GSV_ERR_NO_MEMORY; goto cleanup; }
    gsv_out_f32(&e->voice.enc_p, &eo[0], m0, (size_t)inter * L0);
    gsv_out_f32(&e->voice.enc_p, &eo[1], logs0, (size_t)inter * L0);

    /* ---- speed interpolation of m/logs: L0 -> L1 (torch size=int(L0/speed)+1) ---- */
    if (speed != 1.0f) {
        L1 = (int)((float)L0 / speed) + 1;
        m1 = (float *)malloc(sizeof(float) * (size_t)inter * (size_t)L1);
        logs1 = (float *)malloc(sizeof(float) * (size_t)inter * (size_t)L1);
        if (!m1 || !logs1) { rc = GSV_ERR_NO_MEMORY; goto cleanup; }
        for (int c = 0; c < inter; c++) {
            interp_linear_f32(m0 + (size_t)c * L0, L0, m1 + (size_t)c * L1, L1);
            interp_linear_f32(logs0 + (size_t)c * L0, L0, logs1 + (size_t)c * L1, L1);
        }
    } else {
        L1 = L0; m1 = m0; logs1 = logs0;
    }
    fmask1 = (float *)malloc(sizeof(float) * (size_t)L1);
    if (!fmask1) { rc = GSV_ERR_NO_MEMORY; goto cleanup; }
    for (int t = 0; t < L1; t++) fmask1[t] = 1.0f;

    /* ---- z_p = m + randn * exp(logs) * noise_scale ---- */
    size_t nfeat = (size_t)inter * (size_t)L1;
    zp = (float *)malloc(sizeof(float) * nfeat);
    if (!zp) { rc = GSV_ERR_NO_MEMORY; goto cleanup; }
    if (noise_scale != 0.0f) {
        for (size_t i = 0; i < nfeat; i++)
            zp[i] = m1[i] + gsv_rng_gaussian(e) * expf(logs1[i]) * noise_scale;
    } else {
        memcpy(zp, m1, nfeat * sizeof(float));
    }

    /* ---- flowdec: (z_p [1,inter,L1], y_mask [1,1,L1], ge [1,gin,1]) -> audio ---- */
    int64_t fd[3] = {1, inter, L1};
    int64_t fmd[3] = {1, 1, L1};
    int64_t fgd[3] = {1, gin, 1};
    iz = gsv_in_float(api, &e->voice.flowdec, zp, fd, 3, nfeat, &hz);
    ifm = gsv_in_float(api, &e->voice.flowdec, fmask1, fmd, 3, (size_t)L1, &hfm);
    ifg = gsv_in_float(api, &e->voice.flowdec, ref->ge, fgd, 3, (size_t)gin, &hge2);
    if (!iz || !ifm || !ifg) goto cleanup;
    const void *fin[3] = { iz, ifm, ifg };
    rc = gsv_sess_run(api, &e->voice.flowdec, fin, fo, 1);
    gsv_tensor_destroy(api, iz); gsv_tensor_destroy(api, ifm); gsv_tensor_destroy(api, ifg);
    free(hz); free(hfm); free(hge2); hz=hfm=hge2=NULL; iz=ifm=ifg=NULL;
    if (rc != 0) { gsv_set_error(e, "gsv_sovits_decode: flowdec run failed"); goto cleanup; }

    int n_out = (int)gsv_out_elems(&fo[0]);
    if (n_out > out_capacity) {
        gsv_set_error(e, "gsv_sovits_decode: output capacity %d < %d samples", out_capacity, n_out);
        rc = GSV_ERR_NO_MEMORY; goto cleanup;
    }
    gsv_out_f32(&e->voice.flowdec, &fo[0], out_audio, (size_t)n_out);
    *out_len = n_out;
    rc = GSV_OK;

cleanup:
    gsv_tensor_destroy(api, iq); gsv_tensor_destroy(api, it); gsv_tensor_destroy(api, ig);
    gsv_tensor_destroy(api, iy); gsv_tensor_destroy(api, im); gsv_tensor_destroy(api, iv);
    gsv_tensor_destroy(api, iz); gsv_tensor_destroy(api, ifm); gsv_tensor_destroy(api, ifg);
    free(hq); free(hge); free(hy); free(htm);
    free(hz); free(hfm); free(hge2);
    gsv_sess_free_outputs(api, eo, 3);
    gsv_sess_free_outputs(api, fo, 1);
    free(qp); free(ymask_p); free(txt_ids); free(tmask);
    free(m0); free(logs0);
    if (speed != 1.0f) { free(m1); free(logs1); }   /* else m1==m0 / logs1==logs0 */
    free(fmask1); free(zp);
    return rc;
}
