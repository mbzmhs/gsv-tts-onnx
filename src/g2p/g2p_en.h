/* g2p_en.h - C port of the English G2P orchestration:
 *   GPT_SoVITS/G2P/__init__.py  (text_to_phonemes, "en" branch)
 *   GPT_SoVITS/G2P/English/english.py  (EnglishG2P: text_normalize, g2p, _g2p, qryword)
 * Composes the already-verified components: en_textnorm, tweet_tokenizer,
 * pos_tag_en, en_dicts (cmu/namedict), wseg, gru_predict, homographs_en. */
#ifndef GSV_G2P_EN_H
#define GSV_G2P_EN_H

#include "en_dicts.h"
#include "pos_tag_en.h"
#include "gru_predict.h"
#include "wseg.h"
#include "utf8.h"

/* phones / word2ph result. All strings point into static tables or the handle's
 * arena (valid until the next gsv_g2p_en_* call on the same handle, or free). */
typedef struct {
    char **phones; int n_phones;   /* symbol strings */
    char **words;  int n_words;    /* word2ph["word"]: one entry per token */
    int   *phs;    int n_phs;      /* word2ph["ph"]: phone count per token */
} gsv_g2p_en_result;

typedef struct {
    gsv_en_dict cmu;
    gsv_en_dict namedict;
    gsv_pos_tagger pos;
    gsv_gru gru;
    gsv_wseg wseg;

    gsv_buf normbuf;   /* normalized text scratch (persists through g2p) */
    gsv_buf arena;     /* per-call: homograph split tokens + copied wordsegment comps */

    void *tok_arena;   /* tweet_tokenizer arena from the last g2p call (freed next call) */

    /* malloc'd pointer arrays owned by the handle, rebuilt each g2p call. */
    char **keep_ph; int keep_ph_n, keep_ph_cap;
    char **keep_wd; int keep_wd_n, keep_wd_cap;
    int   *keep_ps; int keep_ps_n, keep_ps_cap;
} gsv_g2p_en_full;

int  gsv_g2p_en_load(gsv_g2p_en_full *g, const char *dir);
void gsv_g2p_en_free(gsv_g2p_en_full *g);

/* english.py text_normalize(text): normalize() + strip non-[a-zA-Z\s pause] + collapse \s+.
 * Result NUL-terminated in g->normbuf. Returns pointer (valid until next call). */
const char *gsv_g2p_en_text_normalize(gsv_g2p_en_full *g, const char *text);

/* english.py g2p(norm_text): tokenize + pos_tag + per-word pron -> filtered phones + word2ph
 * (the eng.g2p output, before __init__ post-processing). Resets arena. */
int gsv_g2p_en_g2p(gsv_g2p_en_full *g, const char *norm_text, gsv_g2p_en_result *out);

/* __init__.py post-processing: pause remap + UNK-map + punctuation dedup + UNK/dedup filter,
 * applied IN PLACE to r (reallocating its arrays). */
void gsv_g2p_en_postprocess(gsv_g2p_en_full *g, gsv_g2p_en_result *r);

/* __init__.py text_to_phonemes(text,"en"): dot-collapse + normalize + g2p + postprocess. */
int gsv_g2p_en_to_phonemes(gsv_g2p_en_full *g, const char *text, gsv_g2p_en_result *out);

#endif
