/* g2p_ja.h - C port of the Japanese G2P orchestration (OpenJTalk base only):
 *   GPT_SoVITS/G2P/Japanese/japanese.py  (JapaneseG2P: symbols_to_japanese,
 *     pyopenjtalk_g2p_prosody, preprocess_jap, g2p)
 * Backed by the OpenJTalk 1.11 C frontend (Mecab + NJD + JPCommon). Uses the
 * stock base output (no Sudachi / nani-ONNX / custom postprocessing). */
#ifndef GSV_G2P_JA_H
#define GSV_G2P_JA_H

#include "utf8.h"

/* Opaque handle owning the OpenJTalk objects + per-call scratch. */
typedef struct gsv_g2p_ja gsv_g2p_ja;

/* phones / word2ph result. All strings point into the handle's result arena
 * (valid until the next gsv_g2p_ja_* call on the same handle, or free). */
typedef struct {
    char **phones; int n_phones;   /* phone tokens (n,a,#,],[ ,^,_,$,?, marks) */
    char **words;  int n_words;    /* word2ph["word"]: one entry per token/char */
    int   *phs;    int n_phs;      /* word2ph["ph"]: phone count per entry */
} gsv_g2p_ja_result;

/* Heap-allocated handle. Returns NULL on failure (bad dicdir). */
gsv_g2p_ja *gsv_g2p_ja_create(const char *dicdir);
void        gsv_g2p_ja_destroy(gsv_g2p_ja *j);

/* japanese.py g2p(norm_text): symbols_to_japanese + lower + per-sentence
 * pyopenjtalk_g2p_prosody + mark insertion. Returns phones + word2ph. */
int  gsv_g2p_ja_g2p(gsv_g2p_ja *j, const char *text, gsv_g2p_ja_result *out);

#endif
