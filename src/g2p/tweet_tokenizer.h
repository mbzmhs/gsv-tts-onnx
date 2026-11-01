/* tweet_tokenizer.h - C port of NLTK TweetTokenizer (nltk.tokenize.casual).
 *
 * Reproduces TweetTokenizer().tokenize(text) exactly: HTML-entity fix, HANG_RE
 * collapse, then leftmost ordered-alternation findall over the REGEXPS list
 * (phone-number variant, match_phone_numbers=True default).
 *
 * Tokens are appended to `out` as a NUL-terminated string array; *count is set.
 */
#ifndef GSV_G2P_TWEET_TOKENIZER_H
#define GSV_G2P_TWEET_TOKENIZER_H

#include <stddef.h>
#include "utf8.h"

/* Tokenize `text` (UTF-8, length `n`). Returns array of pointers into a fresh
 * gsv_buf arena (also returned via *arena) holding NUL-terminated copies.
 * Caller frees *arena with free(). Returns 0 on success, -1 on OOM. */
int gsv_tweet_tokenize(const char *text, size_t n,
                       const char ***tokens, size_t *count,
                       void **arena);

#endif
