#ifndef GSV_JIEBA_H
#define GSV_JIEBA_H

#include <stdint.h>
#include <stddef.h>

/* One output token from posseg lcut: a codepoint range [start,end) into the
 * input array plus its POS tag (pointer into this struct's storage). */
typedef struct {
    uint32_t start, end;      /* codepoint indices, end exclusive */
    const char *tag;          /* NUL-terminated POS tag */
} gsv_seg;

typedef struct {
    /* ---- FREQ + word_tag_tab (from jieba_dict.txt: "word freq tag") ---- */
    uint32_t n_entries;
    void   *entries;          /* jentry* */
    uint8_t *kbuf;            /* key byte arena */
    uint32_t kbuf_len, kbuf_cap;
    uint32_t *ht;             /* open-addressing table: entry index+1, 0=empty */
    uint32_t ht_cap;          /* power of two */
    double total;             /* sum of freqs */
    double logtotal;

    /* ---- tagged HMM (from hmm.bin) ---- */
    uint32_t n_states;
    uint8_t *st_char;         /* [n_states] B/M/E/S */
    char    *st_tag;          /* [n_states][4] tag bytes, NUL-terminated */
    double  *start_p;         /* [n_states] */
    uint32_t *tr_cnt, *tr_off;   /* per-state transition count / offset */
    uint16_t *tr_to;                 /* flat to_id list */
    double   *tr_lp;                 /* flat logp list */
    uint32_t *em_cnt, *em_off;       /* per-state emit count / offset */
    uint32_t *em_cp;                 /* flat codepoint list (sorted) */
    double   *em_lp;                 /* flat logp list */
    uint32_t n_cst;
    uint32_t *cst_cp;                /* sorted by codepoint */
    uint16_t *cst_cnt;
    uint32_t *cst_off;               /* per-entry offset into cst_ids */
    uint16_t *cst_ids;               /* flat state-id list */

    /* ---- plain B/M/E/S HMM (from finalseg.bin), for cut / cut_for_search ----
     * states in sorted-char order: 0=B,1=E,2=M,3=S */
    double   fs_start[4];
    uint32_t fs_tr_cnt[4], fs_tr_off[4];
    uint8_t  *fs_tr_to;              /* flat to-id (0..3) */
    double   *fs_tr_lp;
    uint32_t fs_em_cnt[4], fs_em_off[4];
    uint32_t *fs_em_cp;             /* flat codepoint list (sorted per state) */
    double   *fs_em_lp;
} gsv_jieba;

/* A codepoint range [start,end). Used by cut_for_search output. */
typedef struct { uint32_t start, end; } gsv_range;

/* Load jieba_dict.txt, hmm.bin and finalseg.bin from dir. Returns 0 on success. */
int gsv_jieba_load(gsv_jieba *g, const char *dir);
void gsv_jieba_free(gsv_jieba *g);

/* posseg.lcut over a codepoint array of length n. Writes up to maxsegs tokens;
 * returns the number produced (may exceed maxsegs -> caller must size out). */
int gsv_posseg_lcut(gsv_jieba *g, const uint32_t *cps, int n, gsv_seg *out, int maxsegs);

/* jieba.cut_for_search over a codepoint array of length n. Writes up to `max`
 * subword ranges (in yield order); returns the count produced. */
int gsv_cut_for_search(gsv_jieba *g, const uint32_t *cps, int n, gsv_range *out, int max);

#endif
