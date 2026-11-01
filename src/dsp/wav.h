/* wav.h - RIFF WAV I/O */
#ifndef GSV_WAV_H
#define GSV_WAV_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *data;   /* mono, caller frees with free() */
    int64_t n;
    int sample_rate;
} gsv_wav;

/* Load a WAV file (PCM 16/24/32-bit int or 32-bit float), any channel count
 * (downmixed to mono by mean). Returns 0 on success. */
int gsv_wav_load(const char *path, gsv_wav *out);

/* Write mono s16le WAV. Returns 0 on success. */
int gsv_wav_write_s16(const char *path, const int16_t *data, int64_t n, int sample_rate);

void gsv_wav_free(gsv_wav *w);

#ifdef __cplusplus
}
#endif

#endif
