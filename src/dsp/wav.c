/* wav.c - RIFF WAV I/O */
#include "wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16le(const unsigned char *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

int gsv_wav_load(const char *path, gsv_wav *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f); return -1;
    }

    uint16_t fmt_tag = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    int fmt_found = 0, data_off = 0, data_len = 0;

    while (fread(hdr, 1, 8, f) == 8) {
        uint32_t clen = rd_u32le(hdr + 4);
        if (memcmp(hdr, "fmt ", 4) == 0) {
            unsigned char fh[16];
            if (fread(fh, 1, 16, f) != 16) break;
            fmt_tag = rd_u16le(fh);
            channels = rd_u16le(fh + 2);
            sr = rd_u32le(fh + 4);
            bits = rd_u16le(fh + 14);
            fmt_found = 1;
            uint32_t extra = clen - 16;
            if (extra) fseek(f, (long)extra, SEEK_CUR);
        } else if (memcmp(hdr, "data", 4) == 0) {
            data_off = (int)ftell(f);
            data_len = (int)clen;
            break;
        } else {
            fseek(f, (long)clen + (clen & 1), SEEK_CUR);
        }
    }

    if (!fmt_found || !channels || !sr) { fclose(f); return -1; }

    int bytes_per_sample = bits / 8;
    if (bytes_per_sample < 2) { fclose(f); return -1; }
    int64_t frames = (int64_t)data_len / (bytes_per_sample * channels);
    if (frames <= 0) { fclose(f); return -1; }

    float *mono = (float *)malloc(sizeof(float) * frames);
    if (!mono) { fclose(f); return -1; }

    fseek(f, data_off, SEEK_SET);

    unsigned char *buf = (unsigned char *)malloc((size_t)bytes_per_sample * channels);
    if (!buf) { free(mono); fclose(f); return -1; }

    for (int64_t i = 0; i < frames; i++) {
        double acc = 0.0;
        for (int c = 0; c < channels; c++) {
            if (fread(buf, 1, bytes_per_sample, f) != (size_t)bytes_per_sample) { free(mono); free(buf); fclose(f); return -1; }
            double v;
            if (fmt_tag == 3 && bits == 32) {
                float fv; memcpy(&fv, buf, 4); v = fv;
            } else if (bits == 16) {
                int16_t s; memcpy(&s, buf, 2); v = s / 32768.0;
            } else if (bits == 24) {
                int32_t s = (int32_t)(buf[0] | (buf[1] << 8) | (buf[2] << 16));
                if (s & 0x800000) s |= ~0xFFFFFF;
                v = s / 8388608.0;
            } else if (bits == 32) {
                int32_t s; memcpy(&s, buf, 4); v = s / 2147483648.0;
            } else { free(mono); free(buf); fclose(f); return -1; }
            acc += v;
        }
        mono[i] = (float)(acc / channels);
    }

    free(buf);
    fclose(f);
    out->data = mono;
    out->n = frames;
    out->sample_rate = (int)sr;
    return 0;
}

static void wr_u32le(FILE *f, uint32_t v) {
    unsigned char b[4] = {v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff};
    fwrite(b, 1, 4, f);
}
static void wr_u16le(FILE *f, uint16_t v) {
    unsigned char b[2] = {v & 0xff, (v >> 8) & 0xff};
    fwrite(b, 1, 2, f);
}

int gsv_wav_write_s16(const char *path, const int16_t *data, int64_t n, int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data_bytes = (uint32_t)(n * 2);
    fwrite("RIFF", 1, 4, f); wr_u32le(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); wr_u32le(f, 16);
    wr_u16le(f, 1); wr_u16le(f, 1); wr_u32le(f, sample_rate);
    wr_u32le(f, sample_rate * 2); wr_u16le(f, 2); wr_u16le(f, 16);
    fwrite("data", 1, 4, f); wr_u32le(f, data_bytes);
    fwrite(data, 2, (size_t)n, f);
    fclose(f);
    return 0;
}

void gsv_wav_free(gsv_wav *w) {
    if (w->data) free(w->data);
    w->data = NULL; w->n = 0; w->sample_rate = 0;
}
