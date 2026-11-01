/* ort_sess.c - ORT C API session wrapper */
#include "ort_sess.h"
#include "dsp/f16conv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

void gsv_f32_to_f16(const float *in, uint16_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = gsv_scalar_f32_to_f16(in[i]);
}
void gsv_f16_to_f32(const uint16_t *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = gsv_scalar_f16_to_f32(in[i]);
}

OrtValue *gsv_in_float(const OrtApi *api, const gsv_sess *s, const float *f32,
                       const int64_t *dims, int rank, size_t nfloat, uint16_t **hbuf) {
    if (hbuf) *hbuf = NULL;
    if (s->mdtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        uint16_t *hb = (uint16_t *)malloc(sizeof(uint16_t) * (nfloat ? nfloat : 1));
        if (!hb) return NULL;
        gsv_f32_to_f16(f32, hb, nfloat);
        OrtValue *v = gsv_tensor_create(api, hb, dims, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        if (hbuf) *hbuf = hb;
        return v;   /* on tensor-create failure hb leaks (caller frees via *hbuf path) */
    }
    return gsv_tensor_create(api, f32, dims, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

void gsv_out_f32(const gsv_sess *s, const gsv_out *o, float *dst, size_t n) {
    if (s->mdtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        gsv_f16_to_f32((const uint16_t *)o->data, dst, n);
    else
        memcpy(dst, o->data, n * sizeof(float));
}

#ifdef _WIN32
#include <windows.h>
/* ORTCHAR_T is wchar_t on Windows: convert UTF-8 path to UTF-16 */
static wchar_t *gsv_to_wide(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *w = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!w) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}
#endif

int64_t gsv_out_elems(const gsv_out *o) {
    int64_t n = 1;
    for (int i = 0; i < o->rank; i++) n *= o->dims[i];
    return n;
}

/* names are allocated by ORT's default allocator -> must be freed with AllocatorFree */
static void free_strs(const OrtApi *api, OrtAllocator *alloc, char **a, int n) {
    if (!a) return;
    for (int i = 0; i < n; i++)
        if (a[i]) { if (alloc) api->AllocatorFree(alloc, a[i]); else free(a[i]); }
    free(a);
}

int gsv_sess_create(const OrtApi *api, const OrtEnv *env, const char *path,
                    const char *device, int threads, gsv_sess *out) {
    memset(out, 0, sizeof(*out));
    OrtSessionOptions *so = NULL;
    if (api->CreateSessionOptions(&so) != NULL) return -1;

    if (threads > 0) {
        api->SetIntraOpNumThreads(so, threads);
        api->SetInterOpNumThreads(so, 1);
    }
    if (device && device[0] == 'c' && device[1] == 'u') {
        OrtCUDAProviderOptions co;
        memset(&co, 0, sizeof(co));
        co.device_id = 0;
        co.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        co.gpu_mem_limit = SIZE_MAX;
        co.do_copy_in_default_stream = 1;
        /* if CUDA EP is unavailable ORT will error at CreateSession; caller
         * can retry with "cpu" */
        api->SessionOptionsAppendExecutionProvider_CUDA(so, &co);
    }
#ifdef _WIN32
    wchar_t *wpath = gsv_to_wide(path);
    if (!wpath) { api->ReleaseSessionOptions(so); return -1; }
    OrtStatus *st = api->CreateSession(env, (const ORTCHAR_T *)wpath, so, &out->session);
    free(wpath);
#else
    OrtStatus *st = api->CreateSession(env, path, so, &out->session);
#endif
    api->ReleaseSessionOptions(so);
    if (st) {
        const char *msg = api->GetErrorMessage(st);
        fprintf(stderr, "[gsvtts] CreateSession(%s) failed: %s\n", path, msg ? msg : "?");
        api->ReleaseStatus(st); out->session = NULL; return -1;
    }

    /* enumerate inputs/outputs */
    size_t n_in = 0;
    api->SessionGetInputCount(out->session, &n_in);
    out->in_names = (char **)calloc(n_in ? n_in : 1, sizeof(char *));
    OrtAllocator *alloc = NULL;
    api->GetAllocatorWithDefaultOptions(&alloc);
    out->name_alloc = alloc;
    for (size_t i = 0; i < n_in; i++) {
        char *name = NULL;
        if (api->SessionGetInputName(out->session, i, alloc, &name) != NULL || !name) {
            api->ReleaseSession(out->session);
            free_strs(api, alloc, out->in_names, (int)n_in);
            out->session = NULL; out->in_names = NULL;
            return -1;
        }
        out->in_names[i] = name;
    }
    out->n_in = (int)n_in;

    /* detect float I/O dtype: pure-fp16 models expose at least one FLOAT16 input. */
    out->mdtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    for (size_t i = 0; i < n_in; i++) {
        const OrtTypeInfo *info = NULL;
        if (api->SessionGetInputTypeInfo(out->session, i, &info) != NULL || !info) continue;
        const OrtTensorTypeAndShapeInfo *tti = NULL;
        if (api->CastTypeInfoToTensorInfo(info, &tti) == NULL && tti) {
            ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            if (api->GetTensorElementType(tti, &et) == NULL &&
                et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                out->mdtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        }
        api->ReleaseTypeInfo((OrtTypeInfo *)info);
    }

    size_t n_out = 0;
    api->SessionGetOutputCount(out->session, &n_out);
    out->out_names = (char **)calloc(n_out ? n_out : 1, sizeof(char *));
    for (size_t i = 0; i < n_out; i++) {
        char *name = NULL;
        if (api->SessionGetOutputName(out->session, i, alloc, &name) != NULL || !name) {
            api->ReleaseSession(out->session);
            free_strs(api, alloc, out->in_names, out->n_in);
            free_strs(api, alloc, out->out_names, (int)n_out);
            out->session = NULL; out->in_names = NULL; out->out_names = NULL;
            return -1;
        }
        out->out_names[i] = name;
    }
    out->n_out = (int)n_out;
    return 0;
}

void gsv_sess_destroy(const OrtApi *api, gsv_sess *s) {
    if (!s->session) {
        free_strs(api, s->name_alloc, s->in_names, s->n_in);
        free_strs(api, s->name_alloc, s->out_names, s->n_out);
        return;
    }
    api->ReleaseSession(s->session);
    free_strs(api, s->name_alloc, s->in_names, s->n_in);
    free_strs(api, s->name_alloc, s->out_names, s->n_out);
    s->session = NULL; s->in_names = NULL; s->out_names = NULL;
    s->n_in = 0; s->n_out = 0; s->name_alloc = NULL;
}

static OrtMemoryInfo *gsv_cpu_meminfo(const OrtApi *api) {
    static OrtMemoryInfo *mi = NULL;
    if (!mi) api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi);
    return mi;
}

OrtValue *gsv_tensor_create(const OrtApi *api, const void *data,
                            const int64_t *dims, int rank,
                            ONNXTensorElementDataType type) {
    int64_t elems = 1;
    for (int i = 0; i < rank; i++) elems *= dims[i];
    size_t elem_size = 4;   /* float32 default */
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: elem_size = 2; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   elem_size = 8; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  elem_size = 4; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  elem_size = 2; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   elem_size = 1; break;
        default: elem_size = 4; break;
    }
    size_t bytes = (size_t)elems * elem_size;
    OrtValue *v = NULL;
    OrtStatus *st = api->CreateTensorWithDataAsOrtValue(
        gsv_cpu_meminfo(api), (void *)data, (size_t)bytes, dims, (size_t)rank, type, &v);
    if (st) {
        const char *msg = api->GetErrorMessage(st);
        fprintf(stderr, "[gsvtts] CreateTensorWithDataAsOrtValue failed: %s\n", msg ? msg : "?");
        api->ReleaseStatus(st);
        return NULL;
    }
    return v;
}

void gsv_tensor_destroy(const OrtApi *api, OrtValue *v) {
    if (v) api->ReleaseValue(v);
}

/* NOTE: caller owns the input OrtValues (passed as void* in in_data) and must
 * release them with gsv_tensor_destroy after this call. */
int gsv_sess_run(const OrtApi *api, const gsv_sess *s,
                const void **in_data, gsv_out *outs, int n_out) {
    if (n_out != s->n_out) return -1;
    OrtValue **inputs = (OrtValue **)calloc((size_t)(s->n_in ? s->n_in : 1), sizeof(OrtValue *));
    for (int i = 0; i < s->n_in; i++) inputs[i] = (OrtValue *)in_data[i];

    OrtValue **outputs = (OrtValue **)calloc((size_t)(s->n_out ? s->n_out : 1), sizeof(OrtValue *));
    OrtStatus *st = api->Run(s->session, NULL,
                             (const char * const *)s->in_names,
                             (const OrtValue * const *)inputs, (size_t)s->n_in,
                             (const char * const *)s->out_names, (size_t)s->n_out,
                             outputs);
    if (st) {
        api->ReleaseStatus(st);
        free(inputs); free(outputs);
        return -1;
    }

    for (int i = 0; i < s->n_out; i++) {
        const OrtTensorTypeAndShapeInfo *ti = NULL;
        if (api->GetTensorTypeAndShape(outputs[i], &ti) != NULL || !ti) {
            for (int j = 0; j < s->n_out; j++) if (outputs[j]) api->ReleaseValue(outputs[j]);
            free(inputs); free(outputs);
            return -1;
        }
        size_t rank = 0;
        api->GetDimensionsCount(ti, &rank);
        gsv_out *o = &outs[i];
        o->dims = (int64_t *)malloc(sizeof(int64_t) * (rank ? rank : 1));
        if (rank) api->GetDimensions(ti, o->dims, rank);
        o->rank = (int)rank;
        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        api->GetTensorElementType(ti, &et);
        size_t esz = 4;
        switch (et) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: esz = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   esz = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  esz = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   esz = 1; break;
            default: esz = 4; break;
        }
        int64_t count = gsv_out_elems(o);
        size_t byte_len = (size_t)(count > 0 ? count : 1) * esz;
        o->data = malloc(byte_len);
        const void *ptr = NULL;
        api->GetTensorData(outputs[i], &ptr);
        memcpy(o->data, ptr, byte_len);
        api->ReleaseTensorTypeAndShapeInfo((OrtTensorTypeAndShapeInfo *)ti);
    }

    for (int i = 0; i < s->n_out; i++) api->ReleaseValue(outputs[i]);
    free(inputs); free(outputs);
    return 0;
}

void gsv_sess_free_outputs(const OrtApi *api, gsv_out *outs, int n) {
    (void)api;
    for (int i = 0; i < n; i++) {
        if (outs[i].data) free(outs[i].data);
        if (outs[i].dims) free(outs[i].dims);
        outs[i].data = NULL; outs[i].dims = NULL;
    }
}
