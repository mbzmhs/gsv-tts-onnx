/* g2p_zh.h - C port of the zh G2P orchestration:
 *   GPT_SoVITS/G2P/__init__.py  (text_to_phonemes)
 *   GPT_SoVITS/G2P/Chinese/chinese.py  (ChineseG2P: text_normalize, g2p, _g2p, _merge_erhua)
 * Composes the already-verified components: jieba posseg, pypinyin, tone_sandhi,
 * plus opencpop-strict symbol mapping, erhua, and the __init__ post-processing. */
#ifndef GSV_G2P_ZH_H
#define GSV_G2P_ZH_H

#include "pypinyin.h"
#include "jieba.h"
#include "utf8.h"

/* phones / word2ph result. All strings point into the owner's arena (valid until
 * the next gsv_g2p_* call on the same handle, or free). */
typedef struct {
    char **phones; int n_phones;   /* symbol strings */
    char **words;  int n_words;    /* word2ph["word"]: one entry per char */
    int   *phs;    int n_phs;      /* word2ph["ph"]: phone count per entry */
} gsv_g2p_result;

typedef struct {
    gsv_g2p_zh py;                 /* pinyin data (loaded) */
    gsv_jieba  jb;                 /* jieba posseg data (loaded) */

    /* opencpop-strict map: pinyin -> (initial symbol c, final symbol v). Linear scan. */
    char    **oc_key; uint32_t *oc_klen; char **oc_c; char **oc_v; int n_oc;

    gsv_buf intern;                /* holds all loaded strings (opencpop keys/vals) */
    gsv_buf normbuf;               /* normalized text scratch (persists through g2p) */
    gsv_buf arena;                 /* per-call: result STRINGS (symbols, word chars) */

    /* malloc'd pointer arrays from the last g2p/to_phonemes call (freed on next call).
     * Result strings live in arena; these arrays just hold the pointers. */
    char **keep_ph; char **keep_wd; int *keep_ps;
} gsv_g2p_full;

int  gsv_g2p_full_load(gsv_g2p_full *g, const char *dir);
void gsv_g2p_full_free(gsv_g2p_full *g);

/* chinese.py text_normalize(text): TextNormalizer + keep han(4E00-9FA5)/pause + 嗯/呣.
 * Result NUL-terminated in g->normbuf. Returns pointer (valid until next call). */
const char *gsv_g2p_zh_text_normalize(gsv_g2p_full *g, const char *text);

/* chinese.py g2p(text): split on pause + _g2p per segment -> RAW phones + word2ph
 * (before __init__ post-processing). Resets arena. */
int gsv_g2p_zh_g2p(gsv_g2p_full *g, const char *text, gsv_g2p_result *out);

/* __init__.py post-processing: pause remap + UNK + dedup, applied IN PLACE to r. */
void gsv_g2p_postprocess(gsv_g2p_full *g, gsv_g2p_result *r);

/* __init__.py text_to_phonemes(text,"zh"): dot-collapse + normalize + g2p + postprocess. */
int gsv_g2p_full_to_phonemes(gsv_g2p_full *g, const char *text, gsv_g2p_result *out);

#endif
