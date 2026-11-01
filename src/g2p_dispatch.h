/* g2p_dispatch.h - language-dispatched G2P -> phone ids (M6).
 *
 * Composes the per-language G2P modules (zh / en / ja) into a single entry
 * point that returns the final phoneme id sequence for a piece of text, ready
 * to feed the t2s GPT (x_ids) and the SoVITS decoder (phones2).
 *
 * Faithful to GPT_SoVITS/G2P/__init__.py::text_to_phonemes + phonemes_to_ids:
 *   - zh / en use their verified *_to_phonemes (dot-collapse + normalize + g2p
 *     + the shared pause/UNK/dedup post-processing, already baked in).
 *   - ja uses gsv_g2p_ja_g2p (raw JapaneseG2P output) and then applies the same
 *     shared post-processing here, since the ja module does not bake it in.
 * Phone ids are indices into Symbols.symbols (see gsv_symbol_to_id).
 */
#ifndef GSV_G2P_DISPATCH_H
#define GSV_G2P_DISPATCH_H

#include <stdint.h>
#include "gsvtts.h"

/* Opaque handle owning the zh/en/ja G2P sub-engines. */
typedef struct gsv_g2p_dispatch gsv_g2p_dispatch;

/* Create from a base G2P data dir (e.g. <pkg>/g2p). Loads:
 *   <base>/zh  -> Chinese G2P      (gsv_g2p_full)
 *   <base>/en  -> English G2P      (gsv_g2p_en_full)
 *   <base>/ja  -> Japanese dicdir  (OpenJTalk, gsv_g2p_ja)
 * Missing sub-dirs are tolerated: that language simply becomes unavailable and
 * gsv_g2p_dispatch_ids() returns GSV_ERR_NOT_LOADED for it. Returns NULL only on
 * allocation failure. */
gsv_g2p_dispatch *gsv_g2p_dispatch_create(const char *base_dir);
void              gsv_g2p_dispatch_destroy(gsv_g2p_dispatch *d);

/* Non-zero if the given language ("zh"/"en"/"ja") is loaded. */
int  gsv_g2p_dispatch_has(const gsv_g2p_dispatch *d, const char *lang);

/* Convert text to its final phoneme id sequence.
 * lang: "zh" | "en" | "ja". out_ids must hold at least cap ids; on success
 * *out_n is the count (0 for text with no speakable phones). Returns GSV_OK,
 * GSV_ERR_INVALID_ARG (bad/unsupported lang), GSV_ERR_NOT_LOADED (lang data
 * missing), or GSV_ERR_NO_MEMORY (cap too small). */
int  gsv_g2p_dispatch_ids(const gsv_g2p_dispatch *d, const char *text,
                          const char *lang, int64_t *out_ids, int cap, int *out_n);

/* Full result: final phone ids PLUS the post-processed word2ph (needed for zh
 * BERT alignment). All fields are caller-owned (heap); release with
 * gsv_g2p_full_result_free. For non-zh langs words/phs are NULL/0 (BERT is zeros
 * there anyway) and only ids is populated. */
typedef struct {
    int64_t *ids;  int n_ids;
    char **words;  int n_words;   /* word2ph["word"] (one entry per content char, zh) */
    int *phs;      int n_phs;     /* word2ph["ph"]: phone count per entry (zh) */
} gsv_g2p_full_result;

int  gsv_g2p_dispatch_full(const gsv_g2p_dispatch *d, const char *text, const char *lang,
                           gsv_g2p_full_result *out);
void gsv_g2p_full_result_free(gsv_g2p_full_result *r);

#endif
