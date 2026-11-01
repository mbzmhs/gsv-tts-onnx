/* manifest.h - typed model-package manifest (parsed from manifest.json) */
#ifndef GSV_MANIFEST_H
#define GSV_MANIFEST_H

typedef struct {
    char onnx[128];           /* path relative to pkg root, e.g. "features/bert_int8.onnx" */
    char input_names[8][64];
    int n_in;
    char output_names[8][64];
    int n_out;
    int dim;
    int fp16;                 /* 0=fp32 (default), 1=fp16 */
    int gin_channels;         /* ge graph only */
    char vocab[128];          /* bert only: tokenizer vocab.txt, relative to pkg root */
} gsv_graph_cfg;

typedef struct {
    int max_len;              /* KV buffer length */
    int num_layers, num_heads, model_dim;
    int phoneme_vocab_size;   /* x_ids range */
    int vocab_size;           /* y_ids / logits dim */
    int eos;
    int *suppressed;          /* malloc'd ids */
    int n_suppressed;
} gsv_t2s_cfg;

typedef struct {
    char codebook[128];         /* path relative to pkg root, e.g. "vq/codebook.bin" */
    int dim, bins, n_q;
    int ssl_proj_kernel, ssl_proj_stride;
    char ssl_proj_file[128];    /* weight [dim,dim,k] + bias [dim], fp32; relative to pkg root */
} gsv_vq_cfg;

typedef struct {
    char name[128];
    int inter_channels, hidden_channels, filter_channels;
    int n_heads, n_layers, kernel_size;
    int gin_channels;
    int ly_max, lt_max;       /* static export lengths for enc_p */
    int samples_per_frame;    /* prod(upsample_rates), e.g. 320 */
    int spec_filter_length, spec_hop_length, spec_win_length;
} gsv_voice_cfg;

typedef struct {
    int contract;
    char tts_version[32];     /* short id: "v2pp" (or "v4"); legacy "v2ProPlus" tolerated */
    int sample_rate;          /* output wav rate (32000 v2, 48000 v4) */
    int semantic_hz;          /* 25 */

    /* Per-character model locations (prefix + fallback dir, relative to pkg root).
     * GPT:    <gpt_prefix>/<name>/t2s_{prefill,step}_fp16.onnx        else <gpt_fallback>/...
     * SoVITS: <sovits_prefix>/<name>/{enc_p,flowdec}_<variant>.onnx   else <sovits_fallback>/...
     * The engine does NOT enumerate which names exist — it just plugs `name` into the
     * template and falls back to the base (底模) dir when the per-name file is absent. */
    char gpt_prefix[128];
    char gpt_fallback[128];
    char sovits_prefix[128];
    char sovits_fallback[128];

    gsv_t2s_cfg t2s;
    gsv_graph_cfg bert, hubert, sv, ge;
    gsv_vq_cfg vq;
    gsv_voice_cfg base_voice;  /* version-level SoVITS params (fixed per version) */
} gsv_manifest;

/* Load + parse manifest.json at `path`. Returns 0 on success. */
int gsv_manifest_load(const char *path, gsv_manifest *m);
void gsv_manifest_free(gsv_manifest *m);

#endif
