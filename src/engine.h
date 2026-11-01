/* engine.h - internal engine state (grows as modules land) */
#ifndef GSV_ENGINE_H
#define GSV_ENGINE_H

#include "gsvtts.h"
#include "manifest.h"
#include "ort_sess.h"
#include "bert_tok.h"
#include <onnxruntime_c_api.h>
#include <windows.h>
#include <pthread.h>

struct gsv_g2p_dispatch;   /* fwd (g2p_dispatch.h) */

typedef struct gsv_ref {
    char name[128];
    char wav_path[512];
    char text[4096];
    char lang[8];
    /* prompt_semantic codes, ge embedding: filled in M3 */
    int *prompt_codes;
    int n_prompt_codes;
    float *ge;                /* [gin] (flowdec) */
    int ge_dim;
    float *ge512;             /* [512] (enc_p) */
    /* prompt phoneme ids: G2P of ref text (M6) */
    int64_t *phones1;
    int n_phones1;
    /* prompt BERT features [n_phones1*1024] (M6b, zh only; NULL otherwise) */
    float *bert;
} gsv_ref;

typedef struct gsv_voice {
    char name[128];
    char model_dir[512];      /* <pkg>/voices/<name> */
    gsv_voice_cfg cfg;        /* from manifest voices_cfg */
    gsv_sess enc_p, flowdec;  /* merged encoder (encp_a+proj) + vocoder; dtype per mdtype */
    struct gsv_ref *refs;     /* dynamic array */
    int n_refs;
} gsv_voice;

struct gsv_engine {
    OrtEnv *env;
    const OrtApi *api;
    const OrtApiBase *api_base;
    HMODULE ort_module;

    char device[16];          /* requested: "auto" | "cuda" | "cpu" */
    char active_device[16];   /* ACTUAL device a loaded model ran on ("cuda"/"cpu");
                                * set in load flow so state reports truthfully even when
                                * CUDA was requested but fell back to CPU. */
    int cuda_fallback;        /* 1 if a "cuda" load failed and we retried on CPU */
    int intra_op_threads;     /* 0 = physical cores */
    int use_fp16;             /* fp16 model variant: -1 auto (cuda->*_fp16.onnx if present),
                                 * 0 off (always fp32), 1 on (prefer *_fp16.onnx) */
    char ort_lib_dir[512];

    /* base model package (loaded via gsv_load_base at init from the version manifest) */
    char pkg_dir[512];        /* dir containing the base manifest.json */
    gsv_manifest mf;          /* per-version base manifest; persists across character loads */
    int has_base;             /* 1 once the base set (hubert/sv/ge/bert/vq/g2p + params) is loaded */

    /* base sessions: persist for the engine's lifetime */
    gsv_sess bert, hubert, sv, ge;
    /* per-character GPT sessions (hot-pluggable; reloaded on each gsv_load_model) */
    gsv_sess t2s_prefill, t2s_step;
    float *codebook;          /* [bins*dim] fp32 */
    float *ssl_proj_w;        /* [dim*dim*k] fp32 (prompt-codes Conv1d) */
    float *ssl_proj_b;        /* [dim] fp32 */

    struct gsv_g2p_dispatch *g2p;   /* language G2P dispatcher (M6), from <pkg>/g2p */
    gsv_bert_tok bert_tok;         /* zh RoBERTa WordPiece vocab (M6b) */
    int bert_loaded;               /* 1 once the (zh-only) bert.onnx session is lazily created */
    int bert_failed;               /* 1 if lazy load failed/skipped (don't retry; degrade to zeros) */

    gsv_voice voice;          /* single-resident loaded voice */
    int has_voice;

    gsv_sampling sampling;    /* token sampling + flow noise */

    float sentence_pause;     /* legacy inter-sentence silence, seconds (streaming path only) */
    float cut_mute;           /* chunk-boundary silence base, seconds (÷speed, × tail scale) */
    float cut_minlen;         /* merge clauses until this many semantic chars (CJK=1, en word=1.75) */

    uint64_t rng_state;       /* xorshift64* (token sampling + flow noise) */

    pthread_mutex_t infer_lock;  /* serializes all per-engine ops: shared t2s scratch,
                                   RNG, single-resident voice/refs are not thread-safe */

    /* t2s scratch, lazily allocated on first generate (M4). t2s is pure-fp16, so the KV
     * cache is stored in the model's native fp16 and fed/copied directly (no per-step
     * big-tensor conversion). */
    uint16_t *t2s_kbuf;       /* [L*H*max_len*hd] fp16 */
    uint16_t *t2s_vbuf;       /* [L*H*max_len*hd] fp16 */
    int64_t *t2s_idx;         /* [max_len] arange */

    char last_error[512];
};

/* error helpers (thread-local message, engine-scoped copy) */
void gsv_set_error(gsv_engine *e, const char *fmt, ...);
const char *gsv_global_err_str(void);

/* ORT bootstrap: loads onnxruntime*.dll, returns 0 on success */
int gsv_ort_bootstrap(gsv_engine *e);

/* resolve "auto" device: cuda if a CUDA EP is available, else cpu.
 * fills `out` ("cuda"/"cpu"), returns 0 on success. */
int gsv_resolve_device(const gsv_engine *e, char *out, int n);

/* Load the per-version base set from a manifest path (called at init). Parses +
 * validates the manifest, resolves the device once (cuda->cpu fallback), and loads
 * hubert/sv/ge/bert-vocab/vq/g2p + version params. Sets e->has_base on success. */
int gsv_load_base(gsv_engine *e, const char *manifest_path);

/* Release all loaded models (character + base) — used by gsv_engine_destroy. */
void gsv_engine_release_models(gsv_engine *e);

/* lazily create the zh-only bert.onnx session on first use (P2 memory). Must be
 * called while holding e->infer_lock (all BERT call sites do). 0 on ready. */
int gsv_bert_ensure(gsv_engine *e);

/* fill reference-server defaults (top_k=15, top_p=1.0, temp=1.0, rep=1.35, noise=0.5) */
void gsv_sampling_defaults(gsv_sampling *s);

/* RNG: xorshift64* on e->rng_state (seedable via config "rng_seed"). */
uint64_t gsv_rng_next(gsv_engine *e);
float    gsv_rng_uniform(gsv_engine *e);   /* (0,1) */
float    gsv_rng_gaussian(gsv_engine *e);  /* standard normal (Box-Muller) */

#endif
