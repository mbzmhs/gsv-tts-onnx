/* f16conv.h - IEEE 754 half <-> single scalar converters (pure C, no deps).
 *
 * Used at model I/O boundaries on CUDA: the engine keeps signal processing in fp32 and only
 * converts the tensors that cross into/out of a pure-fp16 graph. Round-to-nearest-even in the
 * normal range (matches numpy.float16 exactly); subnormals (< 2^-14 ~ 6e-5) are flushed toward
 * zero and overflow saturates to +/-inf — both irrelevant for TTS feature magnitudes, which are
 * either ~0 or well within [~1e-3, ~1e3]. */
#ifndef GSV_F16CONV_H
#define GSV_F16CONV_H

#include <stdint.h>
#include <string.h>

/* fp32 -> fp16 (round-to-nearest-even in the normal range). */
static inline uint16_t gsv_scalar_f32_to_f16(float x) {
    uint32_t f; memcpy(&f, &x, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000u;
    uint32_t ef   = (f >> 23) & 0xFFu;      /* fp32 exponent field */
    uint32_t m23  = f & 0x7FFFFFu;
    if (ef == 0xFFu)                          /* inf / NaN */
        return (uint16_t)(sign | 0x7C00u | ((m23 != 0u) ? 0x200u : 0u));
    int32_t e = (int32_t)ef - 127 + 15;      /* rebias to fp16 exponent field */
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);   /* overflow -> inf */
    if (e <= 0) {                              /* zero or subnormal result: flush toward zero */
        if (e < -10) return (uint16_t)sign;     /* underflow -> signed zero */
        uint32_t m = (m23 | 0x800000u) >> (uint32_t)(14 - e);   /* align to 10-bit field */
        if (((m23 | 0x800000u) >> (uint32_t)(13 - e)) & 1u) m += 1u;  /* round dropped bit */
        if (m >= 0x400u) return (uint16_t)(sign | 0x400u);       /* rounded up to smallest normal */
        return (uint16_t)(sign | (m & 0x3FFu));
    }
    uint32_t m10 = m23 >> 13;                  /* top 10 mantissa bits */
    uint32_t low = m23 & 0x1FFFu;              /* dropped low 13 bits */
    if (low > 0x1000u || (low == 0x1000u && (m10 & 1u))) {         /* RNE */
        m10 += 1u;
        if (m10 >= 0x400u) {                    /* carry into the exponent field */
            m10 = 0u; e += 1;
            if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);      /* overflow -> inf */
        }
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | m10);
}

/* fp16 -> fp32 (exact). */
static inline float gsv_scalar_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t ef   = (h >> 10) & 0x1Fu;         /* fp16 exponent field */
    uint32_t m10  = h & 0x3FFu;
    uint32_t f;
    if (ef == 0u) {
        if (m10 == 0u) f = sign;               /* signed zero */
        else {                                   /* subnormal: normalize */
            uint32_t e = 113u;                  /* 127 - 15 + 1 */
            while ((m10 & 0x400u) == 0u) { m10 <<= 1; e--; }
            f = sign | (e << 23) | ((m10 & 0x3FFu) << 13);
        }
    } else if (ef == 0x1Fu) {
        f = sign | 0x7F800000u | (m10 << 13);   /* inf / NaN */
    } else {
        f = sign | ((ef + 112u) << 23) | (m10 << 13);  /* ef - 15 + 127 == ef + 112 */
    }
    float out; memcpy(&out, &f, sizeof(out)); return out;
}

#endif
