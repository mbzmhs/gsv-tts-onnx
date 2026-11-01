/* bert_tok.h - RoBERTa WordPiece tokenizer + per-phone BERT features (M6b, zh).
 *
 * Mirrors GPT_SoVITS/Featurizer/cnroberta.py: tokenize the word2ph text with the
 * chinese-roberta-wwm-ext-large BertPreTokenizer+WordPiece, run bert.onnx (which
 * emits hidden_states[-3]), drop [CLS]/[SEP], then repeat_interleave each content
 * token's feature by its word's phone count -> one 1024-dim vector per phone. */
#ifndef GSV_BERT_TOK_H
#define GSV_BERT_TOK_H

#include <stdint.h>
#include <stddef.h>

struct gsv_engine;   /* fwd (engine.h); avoids a circular include */

typedef struct gsv_bert_tok {
    char **tok;      /* [n] token string per vocab line (in arena) */
    uint32_t *hash;  /* [cap] stored FNV-1a hash per slot */
    int *idx;        /* [cap] slot -> vocab index, -1 = empty */
    int cap;         /* power-of-two capacity */
    int n;           /* vocab size */
    char *arena;     /* owns all token strings */
    int cls, sep, unk;   /* special token ids (looked up at load) */
} gsv_bert_tok;

/* Load a BERT vocab.txt (one token per line). Returns 0 on success. */
int  gsv_bert_tok_load(gsv_bert_tok *t, const char *vocab_path);
void gsv_bert_tok_free(gsv_bert_tok *t);

/* id of `token` in the vocab, or -1 if absent. */
int  gsv_bert_tok_id(const gsv_bert_tok *t, const char *token);

/* Full BertPreTokenizer + WordPiece on `text` -> input_ids incl [CLS]/[SEP].
 * out must hold at least (n_codepoints + 2) ids. Returns 0 and sets *n_tokens. */
int gsv_bert_tokenize(const gsv_bert_tok *t, const char *text, int64_t *out,
                      int cap, int *n_tokens);

/* Per-phone BERT features for zh. `words`/`phs` are the post-processed word2ph
 * (one entry per content char; sum(phs)==n_phones). Runs e->bert and fills
 * out = [n_phones*1024] float32 (row-major, matches gsv_t2s_generate's bert arg).
 * Returns GSV_OK on success; non-zero if the token count does not align with
 * n_words (caller should fall back to zeros) or on any failure. */
int gsv_bert_phone_features(struct gsv_engine *e, char **words, int n_words,
                            const int *phs, int n_phones, float *out);

/* Exported test helper: tokenize `text` with the engine's loaded BERT vocab ->
 * input_ids incl [CLS]/[SEP]. out must hold >= (n_codepoints + 2) ids. */
int gsv_bert_token_ids(struct gsv_engine *e, const char *text, int64_t *out,
                       int cap, int *n_out);

#endif
