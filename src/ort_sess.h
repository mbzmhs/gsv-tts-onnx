/* ort_sess.h - thin ORT C API session wrapper (ORT-allocated outputs) */
#ifndef GSV_ORT_SESS_H
#define GSV_ORT_SESS_H

#include <onnxruntime_c_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gsv_sess {
    OrtSession *session;
    char **in_names;
    int n_in;
    char **out_names;
    int n_out;
    OrtAllocator *name_alloc;   /* owns in/out names (ORT default allocator) */
    int mdtype;                 /* float I/O element type: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                  * (fp32 model) or FLOAT16 (pure-fp16 model). Set at create by
                                  * scanning inputs. Callers build fp16/fp32 OrtValues accordingly
                                  * and convert at the boundary (signal processing stays fp32). */
} gsv_sess;

/* Output tensor (ORT-allocated data, owned by caller until freed). */
typedef struct {
    void *data;
    int64_t *dims;   /* malloc'd */
    int rank;
} gsv_out;

int64_t gsv_out_elems(const gsv_out *o);

/* fp16 <-> fp32 boundary converters (IEEE 754 half, round-to-nearest-even). Used at model
 * I/O boundaries on CUDA: C-side signal processing stays fp32; only the tensors crossing into
 * a pure-fp16 graph are converted. Exact for normal-range values; saturates to +/-inf on
 * overflow and flushes subnormals toward zero (both negligible for TTS feature magnitudes). */
void gsv_f32_to_f16(const float *in, uint16_t *out, size_t n);
void gsv_f16_to_f32(const uint16_t *in, float *out, size_t n);

/* Build an input tensor of session s's float dtype from an fp32 source (nfloat elems).
 * fp32 model -> FLOAT passthrough (no copy); fp16 model -> FLOAT16 with a malloc'd
 * uint16 buffer in *hbuf (caller frees after the Run; NULL for fp32 models). */
OrtValue *gsv_in_float(const OrtApi *api, const gsv_sess *s, const float *f32,
                       const int64_t *dims, int rank, size_t nfloat, uint16_t **hbuf);

/* Copy output o of session s into fp32 dst (n elems), converting from fp16 if the
 * model is pure-fp16. Use for every float output crossing back into C-side DSP. */
void gsv_out_f32(const gsv_sess *s, const gsv_out *o, float *dst, size_t n);

/* Create session from file. device: "cuda" or "cpu". threads: 0 = ORT default. */
int gsv_sess_create(const OrtApi *api, const OrtEnv *env, const char *path,
                    const char *device, int threads, gsv_sess *out);
void gsv_sess_destroy(const OrtApi *api, gsv_sess *s);

/* Run: in_data[i] must match input i's declared shape (batch=1).
 * Outputs are ORT-allocated; free with gsv_sess_free_outputs. */
/* Build a host-memory OrtValue tensor (data NOT copied; must outlive the Run). */
OrtValue *gsv_tensor_create(const OrtApi *api, const void *data,
                            const int64_t *dims, int rank,
                            ONNXTensorElementDataType type);
void gsv_tensor_destroy(const OrtApi *api, OrtValue *v);

/* Run: in_data[i] must be a pre-built OrtValue* (gsv_tensor_create). */
int gsv_sess_run(const OrtApi *api, const gsv_sess *s,
                 const void **in_data, gsv_out *outs, int n_out);
void gsv_sess_free_outputs(const OrtApi *api, gsv_out *outs, int n);

#ifdef __cplusplus
}
#endif

#endif
