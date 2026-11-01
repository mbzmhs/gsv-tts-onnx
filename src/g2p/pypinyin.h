/* pypinyin.h - C port of pypinyin.lazy_pinyin for the two styles ChineseG2P needs:
 *   Style.INITIALS and Style.FINALS_TONE3 (with neutral_tone_with_five=True).
 * Data loaded from <dir>/pinyin.bin and <dir>/phrases.bin (see tools/pack_g2p_zh.py). */
#ifndef GSV_G2P_PYPINYIN_H
#define GSV_G2P_PYPINYIN_H

#include "utf8.h"

#define GSV_PY_STYLE_INITIALS   0
#define GSV_PY_STYLE_FINALS3    1

typedef struct { const char *k; uint32_t kl; uint32_t off; uint16_t nc; } gsv_ph_slot;
typedef struct { const char *p; uint32_t l; } gsv_pre_slot;

typedef struct gsv_g2p_zh {
    /* char pinyin: sorted codepoint -> comma-separated toned candidates */
    uint32_t *py_cp;
    char    **py_val;      /* NUL-terminated, into intern */
    int       n_py;

    /* phrase map: phrase -> per-char first-candidate pinyins (packed in ph_data) */
    gsv_buf   intern;      /* holds all key/value strings */
    gsv_buf   ph_data_holder;  /* packed per-phrase candidate arrays */
    char     *ph_data;     /* == ph_data_holder.data */

    /* open-addressing hash table for phrases: slot = (key ptr,len, data_off, n_chars) */
    gsv_ph_slot *ph_tab;
    size_t ph_tab_sz;

    /* prefix set for mmseg (all prefixes of phrase keys) */
    gsv_pre_slot *pre_tab;
    size_t pre_tab_sz;

    gsv_buf outbuf;        /* scratch for output strings, reset per lazy_pinyin call */
    gsv_buf tonebuf;       /* scratch for fallback toned chars (not in intern), per call */
} gsv_g2p_zh;

/* Load pinyin.bin + phrases.bin from dir. Returns 0 on success. */
int  gsv_g2p_zh_load(gsv_g2p_zh *g, const char *dir);
void gsv_g2p_zh_free(gsv_g2p_zh *g);

/* lazy_pinyin(word, style): fills out_strs[0..*out_n-1] with pointers into
 * g->outbuf. Entry count matches pypinyin.lazy_pinyin (one per char for han,
 * one per non-han block). Returns 0 on success. */
int  gsv_pinyin_lazy(gsv_g2p_zh *g, const char *word, int style, char **out_strs, int *out_n);

#endif
