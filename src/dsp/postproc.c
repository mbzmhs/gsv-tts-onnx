/* postproc.c - server-side audio post-processing chain (port of TTS.py). */
#include "postproc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HEAD_THRESHOLD 0.02f
#define HEAD_FRAME_LEN 512
#define HEAD_HOP 256
#define HEAD_SEARCH_LEN 64000
#define HEAD_MARGIN 3200
#define SPIKE_THRESHOLD 0.3f
#define SPIKE_QUIET 0.05f
#define SPIKE_NEIGHBOR 160

static int head_offset_of(const float *audio, int n) {
    int search_len = n < HEAD_SEARCH_LEN ? n : HEAD_SEARCH_LEN;
    if (search_len < HEAD_FRAME_LEN) return search_len;   /* unfold yields no frames */
    int n_frames = (search_len - HEAD_FRAME_LEN) / HEAD_HOP + 1;
    for (int f = 0; f < n_frames; f++) {
        const float *p = audio + (size_t)f * HEAD_HOP;
        float sum = 0.0f;
        for (int i = 0; i < HEAD_FRAME_LEN; i++) sum += p[i] * p[i];
        float rms = sqrtf(sum / (float)HEAD_FRAME_LEN);
        if (rms > HEAD_THRESHOLD) {
            int off = f * HEAD_HOP - HEAD_MARGIN;
            return off < 0 ? 0 : off;
        }
    }
    return search_len;
}

int gsv_audio_postproc(const float *in, int n_in, int sr,
                       float *out, int out_capacity, int *out_len) {
    if (!in || !out || !out_len || n_in < 0 || sr <= 0) return -1;

    /* 1. head trim */
    int off = head_offset_of(in, n_in);
    const float *a = in + off;
    int n = n_in - off;

    int tail = (int)(0.2f * (float)sr);
    int n_out = n + tail;
    if (n_out > out_capacity) { *out_len = n_out; return -1; }

    /* move trimmed audio into the output buffer (memmove when aliased) */
    if (out == in && off > 0) memmove(out, a, (size_t)n * sizeof(float));
    else memcpy(out, a, (size_t)n * sizeof(float));
    float *b = out;

    /* 2. despike (replicate-padded neighborhood, exact port of F.pad+F.avg_pool1d) */
    if (n >= 2 * SPIKE_NEIGHBOR + 1) {
        int ker = 2 * SPIKE_NEIGHBOR + 1;
        size_t npad = (size_t)n + (size_t)ker - 1;
        float *pad = (float *)malloc(npad * sizeof(float));
        if (!pad) return -1;
        for (size_t j = 0; j < npad; j++) {
            int i = (int)j - SPIKE_NEIGHBOR;
            if (i < 0) i = 0;
            if (i >= n) i = n - 1;
            float v = b[i];
            pad[j] = v < 0.0f ? -v : v;
        }
        float sum = 0.0f;
        for (int k = 0; k < ker; k++) sum += pad[k];
        for (int i = 0; i < n; i++) {
            float absx = b[i] < 0.0f ? -b[i] : b[i];
            float neigh = sum / (float)ker - absx / (float)ker;
            if (absx > SPIKE_THRESHOLD && neigh < SPIKE_QUIET) b[i] = 0.0f;
            if (i + 1 < n) sum += pad[i + ker] - pad[i];
        }
        free(pad);
    }

    /* 3. normalize */
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = b[i] < 0.0f ? -b[i] : b[i];
        if (v > mx) mx = v;
    }
    if (mx > 1.0f) {
        for (int i = 0; i < n; i++) b[i] /= mx;
    }

    /* 4. tail pad */
    memset(b + n, 0, (size_t)tail * sizeof(float));
    *out_len = n_out;
    return 0;
}
