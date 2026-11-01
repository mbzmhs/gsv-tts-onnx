/* gsvtts.h - GSV TTS ONNX inference engine, public C API.
 *
 * Pure-C Windows DLL (gsvtts.dll) for local GSV TTS inference
 * (v2ProPlus first; v4 reserved). Callable from C# / C via P/Invoke.
 *
 * All strings are UTF-8. Audio is mono PCM s16le at the model's
 * sample rate (32000 Hz for v2ProPlus, 48000 Hz for v4; see state JSON).
 *
 * Thread-safety: all mutating + inference operations on a single engine are
 * serialized by an internal lock, so concurrent calls from multiple threads
 * are safe (they queue; no data race). For parallel throughput use one engine
 * per thread instead of sharing one.
 */
#ifndef GSVTTS_H
#define GSVTTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#if defined(GSVTTS_BUILD)
#define GSVTTS_API __declspec(dllexport)
#else
#define GSVTTS_API __declspec(dllimport)
#endif
#else
#define GSVTTS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gsv_engine gsv_engine;
typedef struct gsv_stream gsv_stream;

typedef enum {
    GSV_OK = 0,
    GSV_ERR_GENERIC = -1,
    GSV_ERR_INVALID_ARG = -2,
    GSV_ERR_NOT_LOADED = -3,      /* model not loaded */
    GSV_ERR_REF_MISSING = -4,     /* requested ref not warmed up */
    GSV_ERR_NO_MEMORY = -5,
    GSV_ERR_IO = -6,              /* file read failure */
    GSV_ERR_BACKEND = -7,         /* onnxruntime failure */
    GSV_ERR_TEXT_EMPTY = -8,      /* no speakable text after g2p */
    GSV_ERR_BUSY = -9,            /* another inference in flight */
} gsv_status;

/* gsv_stream_next return codes (positive = not an error) */
#define GSV_STREAM_DONE 1         /* no more audio; stream may be ended */

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* Create engine. config_json may be NULL. Recognized keys:
 *   "device": "auto" | "cuda" | "cpu"          (default "auto")
 *   "intra_op_num_threads": int                (0 = physical cores)
 *   "ort_lib_dir":  string                     (dir holding onnxruntime*.dll,
 *                                               default: same dir as gsvtts.dll)
 *   "base_manifest": string                    (path to the per-version base manifest.json;
 *                                               loads the base model set once at init.
 *                                               Omit for a bare engine (no models).)
 * Returns NULL on failure (see gsv_global_last_error()). */
GSVTTS_API gsv_engine *gsv_engine_create(const char *config_json);
GSVTTS_API void        gsv_engine_destroy(gsv_engine *e);

/* Last error message for this engine ("" if none). */
GSVTTS_API const char *gsv_last_error(const gsv_engine *e);
/* Global last error (for create failures before an engine exists). */
GSVTTS_API const char *gsv_global_last_error(void);

/* Library version string, e.g. "0.2.0". Never NULL. */
GSVTTS_API const char *gsv_version(void);

/* ------------------------------------------------------------------ */
/* Model lifecycle                                                     */
/* ------------------------------------------------------------------ */

/* Load a character (SoVITS + GPT) by two INDEPENDENT directory names. The base set
 * must already be loaded via "base_manifest" at create time. Each side resolves
 * against the manifest separately:
 *   GPT:    <gpt_prefix>/<gpt_name>/t2s_{prefill,step}_fp16.onnx  (else gpt.fallback)
 *   SoVITS: <sovits_prefix>/<sovits_name>/{enc_p,flowdec}_<variant>.onnx (else sovits.fallback)
 * An empty name (or a missing per-name directory) falls back to that side's base
 * model, so any SoVITS×GPT combination works without in-package symlinks.
 * Loading a character unloads any previously loaded one (single-resident). */
GSVTTS_API int gsv_load_model(gsv_engine *e, const char *sovits_name, const char *gpt_name);
/* Release the loaded character: cached refs + its GPT/SoVITS sessions. Base persists. */
GSVTTS_API int gsv_unload_model(gsv_engine *e);

/* ------------------------------------------------------------------ */
/* Reference (emotion preset) cache                                    */
/* ------------------------------------------------------------------ */

/* Warm up a single ref: computes prompt semantic codes + global emotion
 * embedding from wav_path (any-rate WAV) and its transcript text/lang.
 * ref_name is the logical key used later in inference calls. */
GSVTTS_API int gsv_warmup_ref(gsv_engine *e, const char *ref_name,
                              const char *wav_path, const char *text, const char *lang);

/* Warm up several refs at once (parallel arrays of length n). */
GSVTTS_API int gsv_warmup_refs(gsv_engine *e,
                                const char *const *ref_names,
                                const char *const *wav_paths,
                                const char *const *texts,
                                const char *const *langs, int n);

/* Drop one cached ref. */
GSVTTS_API int gsv_unload_ref(gsv_engine *e, const char *ref_name);

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

/* Fill buf with a JSON state snapshot:
 *   {"version":..., "device":"cuda|cpu", "sample_rate":32000,
 *    "loaded_voice":"my_voice", "refs":["happy","sad",...],
 *    "model_dir":"...", "ort_version":"1.23.2"}
 * If buf_size < required, returns GSV_OK with *required set and buf untouched. */
 GSVTTS_API int gsv_get_state(const gsv_engine *e, char *buf, int buf_size,
                              int *required);

/* Comma-joined list of ORT execution providers available in this process, e.g.
 * "CPUExecutionProvider,CUDAExecutionProvider". No model load / no VRAM needed. */
 GSVTTS_API int gsv_available_providers(const gsv_engine *e, char *buf, int buf_size);


/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */

/* Non-streaming synthesis of the full text (internal sentence chunking).
 * speed: 1.0 = normal; <1 slower/longer, >1 faster.
 * ref_name: key of a warmed-up ref (emotion).
 * Writes up to out_capacity samples of s16le mono PCM; *written_samples is
 * the total produced. If out_capacity is too small the result is truncated
 * and GSV_OK is still returned with *written_samples = full length. */
GSVTTS_API int gsv_infer(gsv_engine *e,
                          const char *text, const char *lang, float speed,
                          const char *ref_name,
                          int16_t *out_pcm, int64_t out_capacity,
                          int64_t *written_samples);

/* Streaming synthesis (pull-based). begin returns NULL on failure.
 * Repeatedly call next until it returns GSV_STREAM_DONE, then end.
 * Each next() yields the next decoded chunk (sentence granularity). */
GSVTTS_API gsv_stream *gsv_stream_begin(gsv_engine *e,
                                         const char *text, const char *lang,
                                         float speed, const char *ref_name);
/* Returns GSV_OK with *written>0, GSV_STREAM_DONE when finished, or a
 * negative gsv_status on error. */
GSVTTS_API int gsv_stream_next(gsv_stream *s, int16_t *out_pcm,
                               int64_t out_capacity, int64_t *written);
/* Release stream resources (safe to call after DONE). */
GSVTTS_API void gsv_stream_end(gsv_stream *s);

/* ------------------------------------------------------------------ */
/* Sampling parameters                                                 */
/* ------------------------------------------------------------------ */

/* GPT token sampling + flow noise. Defaults match the reference server:
 * top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
 * noise_scale=0.5. Pass NULL for a field to keep its current value. */
typedef struct {
    int   top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    float noise_scale;        /* flow z = m + randn * exp(logs) * noise_scale */
} gsv_sampling;

GSVTTS_API int gsv_set_sampling(gsv_engine *e, const gsv_sampling *s);
GSVTTS_API int gsv_get_sampling(const gsv_engine *e, gsv_sampling *s);

/* Inter-sentence silence (seconds) inserted between sentences during multi-sentence
 * synthesis (long-text gsv_infer fallback + streaming). Default 0.3. Only affects
 * output with >1 sentence; single-shot / single-sentence output is unchanged. */
GSVTTS_API int gsv_set_sentence_pause(gsv_engine *e, float seconds);
GSVTTS_API int gsv_get_sentence_pause(const gsv_engine *e, float *seconds);

/* Chunked-synthesis punctuation pauses (reference-implementation style; applies to every
 * gsv_infer call). Text is cut at sentence AND clause punctuation (，,：:、~・,
 * delimiter kept with its chunk; ellipsis is NOT a cut point — the model renders
 * its pause natively); clauses are merged until the chunk reaches
 * cut_minlen semantic chars (CJK = 1, ASCII alnum run = 1.75). Real silence of
 * cut_mute / speed × tail-scale is appended after every chunk (sentence puncts
 * and … 1.5, 、・ 0.8, else 1.0) and a fixed 0.2s tail ends the output.
 * Defaults: cut_mute 0.3s, cut_minlen 0 (cut at every punctuation mark). */
GSVTTS_API int gsv_set_cut_mute(gsv_engine *e, float seconds);
GSVTTS_API int gsv_get_cut_mute(const gsv_engine *e, float *seconds);
GSVTTS_API int gsv_set_cut_minlen(gsv_engine *e, float chars);
GSVTTS_API int gsv_get_cut_minlen(const gsv_engine *e, float *chars);

/* fp16 model-variant preference (applies at gsv_load_model time). mode: -1 auto
 * (on CUDA, load a <stem>_fp16.onnx sibling when present, else fp32), 0 off (always
 * fp32), 1 on (prefer the *_fp16.onnx variant). fp16 models keep fp32 I/O, so no other
 * API/ABI changes. Default -1. */
GSVTTS_API int gsv_set_fp16(gsv_engine *e, int mode);

/* text -> final phoneme id sequence (same G2P path as gsv_infer). lang "zh"/"en"/"ja".
 * out_ids must hold >= cap ids; on success *out_n is the count. Additive: for callers
 * that want phonemes without full synthesis, and for G2P cross-checks. */
GSVTTS_API int gsv_text_to_phoneme_ids(gsv_engine *e, const char *text, const char *lang,
                                        int64_t *out_ids, int cap, int *out_n);

#ifdef __cplusplus
}
#endif

#endif /* GSVTTS_H */
