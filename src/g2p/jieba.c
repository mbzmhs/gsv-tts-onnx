/* jieba.c - pure-C port of jieba.posseg (DAG DP + tagged-HMM Viterbi lcut).
 * Mirrors runtime/Lib/site-packages/jieba/{__init__,posseg/__init__,posseg/viterbi}.py
 * so that gsv_posseg_lcut reproduces jieba.posseg.lcut byte-for-byte. */
#include "jieba.h"
#include "utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MIN_FLOAT (-3.14e100)
#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif
#define MIN_INF   (-INFINITY)

#define FNV_OFF 2166136261u
#define FNV_PRIME 16777619u

typedef struct { uint32_t koff; uint16_t klen; int64_t freq; char tag[8]; } jentry;

/* ---------------- char class helpers ---------------- */
static int is_cjk(uint32_t c){ return (c>=0x4E00 && c<=0x9FD5); }
static int is_alnum(uint32_t c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9'); }
static int is_dotdigit(uint32_t c){ return c=='.'||(c>='0'&&c<='9'); }
static int is_ws(uint32_t c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
static int is_han_internal(uint32_t c){
    return is_cjk(c)||is_alnum(c)||c=='+'||c=='#'||c=='&'||c=='.'||c=='_';
}

/* ---------------- FNV-1a (incremental) ---------------- */
static uint32_t fnv_ext(uint32_t h, const uint8_t *p, int n){ for(int i=0;i<n;i++){h^=p[i];h*=FNV_PRIME;} return h; }

/* ---------------- FREQ / word_tag_tab lookup ---------------- */
static const jentry *jlookup(gsv_jieba *g, const uint8_t *s, uint16_t len){
    uint32_t h = fnv_ext(FNV_OFF, s, (int)len);
    uint32_t mask = g->ht_cap - 1;
    uint32_t idx = h & mask;
    for(;;){
        uint32_t v = g->ht[idx];
        if(!v) return NULL;
        const jentry *e = (const jentry*)g->entries + (v-1);
        if(e->klen==len && memcmp((const uint8_t*)g->kbuf + e->koff, s, len)==0) return e;
        idx = (idx+1) & mask;
    }
}

static const jentry *range_lookup(gsv_jieba *g, const uint32_t *cps, int a, int b){
    uint8_t frag[1024]; int flen=0;
    for(int k=a;k<b;k++) flen += gsv_utf8_encode(cps[k], (char*)frag+flen);
    return jlookup(g, frag, (uint16_t)flen);
}
static const char *word_tag(gsv_jieba *g, const uint32_t *cps, int a, int b){
    const jentry *e = range_lookup(g, cps, a, b);
    return e ? e->tag : "x";
}

/* ---------------- HMM row lookups (binary search) ---------------- */
static double emit_get(gsv_jieba *g, int y, uint32_t cp){
    uint32_t o=g->em_off[y], c=g->em_cnt[y]; uint32_t lo=0, hi=c;
    while(lo<hi){ uint32_t mid=(lo+hi)>>1; if(g->em_cp[o+mid]<cp) lo=mid+1; else hi=mid; }
    return (lo<c && g->em_cp[o+lo]==cp) ? g->em_lp[o+lo] : MIN_FLOAT;
}
static double trans_get(gsv_jieba *g, int y0, int y){
    uint32_t o=g->tr_off[y0], c=g->tr_cnt[y0]; uint32_t lo=0, hi=c;
    while(lo<hi){ uint32_t mid=(lo+hi)>>1; if(g->tr_to[o+mid]<(uint16_t)y) lo=mid+1; else hi=mid; }
    return (lo<c && g->tr_to[o+lo]==(uint16_t)y) ? g->tr_lp[o+lo] : MIN_INF;
}
static void cst_lookup(gsv_jieba *g, uint32_t cp, const uint16_t **ids, int *n, int *found){
    uint32_t lo=0, hi=g->n_cst;
    while(lo<hi){ uint32_t mid=(lo+hi)>>1; if(g->cst_cp[mid]<cp) lo=mid+1; else hi=mid; }
    if(lo<g->n_cst && g->cst_cp[lo]==cp){ *ids=g->cst_ids + g->cst_off[lo]; *n=g->cst_cnt[lo]; *found=1; }
    else { *ids=NULL; *n=0; *found=0; }
}

/* ---------------- Viterbi (tagged HMM) ---------------- */
static double viterbi(gsv_jieba *g, const uint32_t *obs, int L, int16_t *route){
    int S = g->n_states;
    double *Vprev = malloc((size_t)S*sizeof(double));
    double *Vcur  = malloc((size_t)S*sizeof(double));
    int16_t *memhist = malloc((size_t)L*S*sizeof(int16_t));
    int *active_prev = malloc((size_t)S*sizeof(int));
    int *active_cur  = malloc((size_t)S*sizeof(int));
    int *prev_states = malloc((size_t)S*sizeof(int));
    int *cand        = malloc((size_t)S*sizeof(int));
    uint8_t *in_expect = malloc(S);

    const uint16_t *ids; int nids, have;
    cst_lookup(g, obs[0], &ids, &nids, &have);
    int na_prev=0;
    if(have){ for(int i=0;i<nids;i++){ int y=ids[i]; Vprev[y]=g->start_p[y]+emit_get(g,y,obs[0]); memhist[y]=-1; active_prev[na_prev++]=y; } }
    else    { for(int y=0;y<S;y++){ Vprev[y]=g->start_p[y]+emit_get(g,y,obs[0]); memhist[y]=-1; active_prev[na_prev++]=y; } }

    int na_cur=0;
    for(int t=1;t<L;t++){
        int np=0;
        for(int i=0;i<na_prev;i++){ int x=active_prev[i]; if(g->tr_cnt[x]>0) prev_states[np++]=x; }
        memset(in_expect, 0, (size_t)S);
        for(int i=0;i<np;i++){ int x=prev_states[i]; uint32_t o=g->tr_off[x], c=g->tr_cnt[x];
            for(uint32_t k=0;k<c;k++) in_expect[g->tr_to[o+k]]=1; }

        cst_lookup(g, obs[t], &ids, &nids, &have);
        int ncand=0;
        if(have){ for(int i=0;i<nids;i++){ int y=ids[i]; if(in_expect[y]) cand[ncand++]=y; } }
        else    { for(int y=0;y<S;y++) if(in_expect[y]) cand[ncand++]=y; }
        if(ncand==0){
            ncand=0; int any=0;
            for(int y=0;y<S;y++) if(in_expect[y]){ cand[ncand++]=y; any=1; }
            if(!any) for(int y=0;y<S;y++) cand[ncand++]=y;
        }

        na_cur=0;
        for(int ci=0;ci<ncand;ci++){
            int y=cand[ci]; double bestp=MIN_INF; int bests=-1;
            for(int pi=0;pi<np;pi++){
                int y0=prev_states[pi];
                double val = Vprev[y0] + trans_get(g,y0,y) + emit_get(g,y,obs[t]);
                if(bests<0 || val>bestp || (val==bestp && y0>bests)){ bestp=val; bests=y0; }
            }
            if(bests<0){ bestp=MIN_FLOAT; bests=0; }   /* np==0 (unreachable) guard */
            Vcur[y]=bestp; memhist[(size_t)t*S+y]=(int16_t)bests; active_cur[na_cur++]=y;
        }
        double *tv=Vprev; Vprev=Vcur; Vcur=tv;
        int *ta=active_prev; active_prev=active_cur; active_cur=ta;
        na_prev=na_cur; na_cur=0;
    }

    double bestp=MIN_INF; int bests=-1;
    for(int i=0;i<na_prev;i++){ int y=active_prev[i]; double val=Vprev[y];
        if(bests<0 || val>bestp || (val==bestp && y>bests)){ bestp=val; bests=y; } }
    route[L-1]=(int16_t)bests;
    int st=bests;
    for(int i=L-2;i>=0;i--){ st=memhist[(size_t)(i+1)*S+st]; route[i]=(int16_t)st; }

    free(Vprev); free(Vcur); free(memhist); free(active_prev); free(active_cur);
    free(prev_states); free(cand); free(in_expect);
    return bestp;
}

/* ---------------- token emission ---------------- */
static void emit_tok(gsv_seg *out, int *outn, int max, uint32_t s, uint32_t e, const char *tag){
    if(*outn < max){ out[*outn].start=s; out[*outn].end=e; out[*outn].tag=tag; (*outn)++; }
}

/* __cut: BMES route -> words (HMM on CJK range [a,b)) */
static void cut_hmm(gsv_jieba *g, const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max){
    int L=b-a;
    int16_t *route = malloc((size_t)L*sizeof(int16_t));
    viterbi(g, cps+a, L, route);
    uint32_t begin=(uint32_t)a, nexti=(uint32_t)a;
    for(int i=0;i<L;i++){
        char sc=(char)g->st_char[route[i]];
        if(sc=='B') begin=(uint32_t)(a+i);
        else if(sc=='E'){ emit_tok(out,outn,max,begin,(uint32_t)(a+i+1), g->st_tag + 4*route[i]); nexti=(uint32_t)(a+i+1); }
        else if(sc=='S'){ emit_tok(out,outn,max,(uint32_t)(a+i),(uint32_t)(a+i+1), g->st_tag + 4*route[i]); nexti=(uint32_t)(a+i+1); }
    }
    if(nexti<(uint32_t)b) emit_tok(out,outn,max,nexti,(uint32_t)b, g->st_tag + 4*route[nexti-a]);
    free(route);
}

/* re_skip_detail.split classification for a non-CJK range [a,b) (num/eng/x) */
static void cut_nonhan_detail(const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max){
    int i=a;
    while(i<b){
        uint32_t c=cps[i];
        if(is_dotdigit(c)){ int j=i; while(j<b&&is_dotdigit(cps[j]))j++; emit_tok(out,outn,max,(uint32_t)i,(uint32_t)j,"m"); i=j; }
        else if(is_alnum(c)){ int j=i; while(j<b&&is_alnum(cps[j]))j++; emit_tok(out,outn,max,(uint32_t)i,(uint32_t)j,"eng"); i=j; }
        else { int j=i; while(j<b&&!is_dotdigit(cps[j])&&!is_alnum(cps[j]))j++; emit_tok(out,outn,max,(uint32_t)i,(uint32_t)j,"x"); i=j; }
    }
}

/* __cut_detail: split range [a,b) on CJK runs; HMM on CJK, classify the rest */
static void cut_detail(gsv_jieba *g, const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max){
    int i=a;
    while(i<b){
        if(is_cjk(cps[i])){ int j=i; while(j<b&&is_cjk(cps[j]))j++; cut_hmm(g,cps,i,j,out,outn,max); i=j; }
        else { int j=i; while(j<b&&!is_cjk(cps[j]))j++; cut_nonhan_detail(cps,i,j,out,outn,max); i=j; }
    }
}

/* ---------------- finalseg: plain B/M/E/S HMM ----------------
 * state ids in sorted-char order: 0=B,1=E,2=M,3=S.
 * PrevStatus (allowed previous states for current y):
 *   B<-{E,S}  E<-{B,M}  M<-{M,B}  S<-{S,E} */
static const uint8_t FS_PREV[4][2] = { {1,3},{0,2},{2,0},{3,1} };
static const char    FS_CHAR[4]    = { 'B','E','M','S' };

static double fs_emit(gsv_jieba *g, int y, uint32_t cp){
    uint32_t o=g->fs_em_off[y], c=g->fs_em_cnt[y]; uint32_t lo=0, hi=c;
    while(lo<hi){ uint32_t mid=(lo+hi)>>1; if(g->fs_em_cp[o+mid]<cp) lo=mid+1; else hi=mid; }
    return (lo<c && g->fs_em_cp[o+lo]==cp) ? g->fs_em_lp[o+lo] : MIN_FLOAT;
}
static double fs_trans(gsv_jieba *g, int y0, int y){
    uint32_t o=g->fs_tr_off[y0], c=g->fs_tr_cnt[y0];
    for(uint32_t k=0;k<c;k++) if(g->fs_tr_to[o+k]==(uint8_t)y) return g->fs_tr_lp[o+k];
    return MIN_INF;
}

static void fs_viterbi(gsv_jieba *g, const uint32_t *obs, int L, uint8_t *route){
    double Vprev[4], Vcur[4];
    uint8_t *bp = malloc((size_t)L*4);
    for(int y=0;y<4;y++){ Vprev[y]=g->fs_start[y]+fs_emit(g,y,obs[0]); bp[y]=0xFF; }
    for(int t=1;t<L;t++){
        for(int y=0;y<4;y++){
            double em=fs_emit(g,y,obs[t]); double bestp=MIN_INF; int bests=-1;
            for(int k=0;k<2;k++){
                int y0=FS_PREV[y][k];
                double val=Vprev[y0]+fs_trans(g,y0,y)+em;
                if(bests<0 || val>bestp || (val==bestp && y0>bests)){ bestp=val; bests=y0; }
            }
            Vcur[y]=bestp; bp[(size_t)t*4+y]=(uint8_t)bests;
        }
        for(int y=0;y<4;y++) Vprev[y]=Vcur[y];
    }
    int bests=-1; double bestp=MIN_INF;
    { int cand[2]={1,3};   /* final max over 'ES' */
      for(int i=0;i<2;i++){ int y=cand[i]; double val=Vprev[y];
        if(bests<0 || val>bestp || (val==bestp && y>bests)){ bestp=val; bests=y; } } }
    route[L-1]=(uint8_t)bests;
    int st=bests;
    for(int i=L-2;i>=0;i--){ st=bp[(size_t)(i+1)*4+st]; route[i]=(uint8_t)st; }
    free(bp);
}

/* finalseg.__cut on a pure-CJK block [a,b): BMES route -> words */
static void fs_cut_blk(gsv_jieba *g, const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max){
    int L=b-a; if(L<=0) return;
    uint8_t *route=malloc((size_t)L);
    fs_viterbi(g, cps+a, L, route);
    uint32_t begin=(uint32_t)a, nexti=(uint32_t)a;
    for(int i=0;i<L;i++){
        char sc=FS_CHAR[route[i]];
        if(sc=='B') begin=(uint32_t)(a+i);
        else if(sc=='E'){ emit_tok(out,outn,max,begin,(uint32_t)(a+i+1),"x"); nexti=(uint32_t)(a+i+1); }
        else if(sc=='S'){ emit_tok(out,outn,max,(uint32_t)(a+i),(uint32_t)(a+i+1),"x"); nexti=(uint32_t)(a+i+1); }
    }
    if(nexti<(uint32_t)b) emit_tok(out,outn,max,nexti,(uint32_t)b,"x");
    free(route);
}

/* re_skip.split emulation on a non-CJK run [a,b): pattern [a-zA-Z0-9]+(\.\d+)?%? */
static void fs_cut_nonhan(const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max){
    int i=a, sep=-1;
    while(i<b){
        if(is_alnum(cps[i])){
            if(sep>=0){ emit_tok(out,outn,max,(uint32_t)sep,(uint32_t)i,"x"); sep=-1; }
            int j=i; while(j<b && is_alnum(cps[j])) j++;
            if(j<b && cps[j]=='.'){ int k=j+1; if(k<b && (cps[k]>='0'&&cps[k]<='9')){ while(k<b&&cps[k]>='0'&&cps[k]<='9')k++; j=k; } }
            if(j<b && cps[j]=='%') j++;
            emit_tok(out,outn,max,(uint32_t)i,(uint32_t)j,"eng");
            i=j;
        } else { if(sep<0) sep=i; i++; }
    }
    if(sep>=0) emit_tok(out,outn,max,(uint32_t)sep,(uint32_t)b,"x");
}

/* finalseg.cut on range [a,b): CJK runs -> viterbi, rest -> re_skip */
static void finalseg_cut(gsv_jieba *g, const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max){
    int i=a;
    while(i<b){
        if(is_cjk(cps[i])){ int j=i; while(j<b && is_cjk(cps[j])) j++; fs_cut_blk(g,cps,i,j,out,outn,max); i=j; }
        else { int j=i; while(j<b && !is_cjk(cps[j])) j++; fs_cut_nonhan(cps,i,j,out,outn,max); i=j; }
    }
}

/* flush an accumulated single-char buffer range [a,b).
 * plain=0 -> unrecognized run goes to tagged-HMM (posseg __cut_detail)
 * plain=1 -> unrecognized run goes to finalseg.cut (plain jieba.__cut_DAG) */
static void flush_buf(gsv_jieba *g, const uint32_t *cps, int a, int b, gsv_seg *out, int *outn, int max, int plain){
    if(b-a==1){ emit_tok(out,outn,max,(uint32_t)a,(uint32_t)b, word_tag(g,cps,a,b)); }
    else {
        const jentry *e = range_lookup(g, cps, a, b);
        if(!e || e->freq==0){
            if(plain) finalseg_cut(g, cps, a, b, out,outn,max);
            else      cut_detail(g, cps, a, b, out,outn,max);
        }
        else { for(int k=a;k<b;k++) emit_tok(out,outn,max,(uint32_t)k,(uint32_t)(k+1), word_tag(g,cps,k,k+1)); }
    }
}

/* __cut_DAG (HMM=True path): DP over FREQ on han-internal block [s,e) then walk */
static void cut_dag(gsv_jieba *g, const uint32_t *cps, int s, int e, gsv_seg *out, int *outn, int max, int plain){
    int L=e-s;
    double *rval = malloc((size_t)(L+1)*sizeof(double));
    int *rx = malloc((size_t)(L+1)*sizeof(int));   /* rx[rel] = absolute best end x */
    uint8_t *frag = malloc((size_t)(L+1)*4);
    rval[L]=0.0;
    for(int idx=L-1;idx>=0;idx--){
        int ai=s+idx; double bestv=MIN_INF; int bestx=ai; int found=0;
        int flen=0; uint32_t h=FNV_OFF;
        for(int x=idx;x<L;x++){
            int ax=s+x;
            int e8=gsv_utf8_encode(cps[ax], (char*)frag+flen); flen+=e8;
            h=fnv_ext(h, frag+flen-e8, e8);
            const jentry *ej=jlookup(g, frag, (uint16_t)flen);
            if(!ej) break;
            if(ej->freq!=0){ found=1; double val=log((double)ej->freq)-g->logtotal+rval[x+1];
                if(val>bestv||(val==bestv&&ax>bestx)){ bestv=val; bestx=ax; } }
        }
        if(!found){ bestv=-g->logtotal+rval[idx+1]; bestx=ai; }
        rval[idx]=bestv; rx[idx]=bestx;
    }
    int x=s, bstart=s, hasbuf=0;
    while(x<e){
        int y=rx[x-s]+1;
        if(y-x==1){ if(!hasbuf){ bstart=x; hasbuf=1; } }
        else {
            if(hasbuf){ flush_buf(g,cps,bstart,x,out,outn,max,plain); hasbuf=0; }
            emit_tok(out,outn,max,(uint32_t)x,(uint32_t)y, word_tag(g,cps,x,y));
        }
        x=y;
    }
    if(hasbuf) flush_buf(g,cps,bstart,e,out,outn,max,plain);
    free(rval); free(rx); free(frag);
}

/* __cut_internal else-branch: non han-internal block [s,e) */
static void cut_nonhan_block(const uint32_t *cps, int s, int e, gsv_seg *out, int *outn, int max){
    int i=s;
    while(i<e){
        uint32_t c=cps[i];
        if(is_ws(c)){
            int len=1;
            if(c=='\r' && i+1<e && cps[i+1]=='\n') len=2;
            emit_tok(out,outn,max,(uint32_t)i,(uint32_t)(i+len),"x");   /* whole skip run */
            i+=len;
        } else {
            int j=i; while(j<e && !is_ws(cps[j])) j++;
            int x_alnum = is_alnum(cps[i]);   /* re_eng.match(x) on the whole run */
            for(int k=i;k<j;k++){
                uint32_t cc=cps[k];
                if(is_dotdigit(cc)) emit_tok(out,outn,max,(uint32_t)k,(uint32_t)(k+1),"m");
                else if(x_alnum)    emit_tok(out,outn,max,(uint32_t)k,(uint32_t)(k+1),"eng");
                else                emit_tok(out,outn,max,(uint32_t)k,(uint32_t)(k+1),"x");
            }
            i=j;
        }
    }
}

/* __cut_internal (HMM=True) */
int gsv_posseg_lcut(gsv_jieba *g, const uint32_t *cps, int n, gsv_seg *out, int maxsegs){
    int outn=0;
    int i=0;
    while(i<n){
        if(is_han_internal(cps[i])){ int j=i; while(j<n&&is_han_internal(cps[j]))j++; cut_dag(g,cps,i,j,out,&outn,maxsegs,0); i=j; }
        else { int j=i; while(j<n&&!is_han_internal(cps[j]))j++; cut_nonhan_block(cps,i,j,out,&outn,maxsegs); i=j; }
    }
    return outn;
}

/* jieba.cut_for_search: plain cut then expand each word with 2/3-grams in FREQ */
int gsv_cut_for_search(gsv_jieba *g, const uint32_t *cps, int n, gsv_range *out, int max){
    if(n<=0) return 0;
    int cap = 2*n + 8;
    gsv_seg *words = malloc(sizeof(gsv_seg)*cap);
    int wn=0;
    int i=0;
    while(i<n){
        if(is_han_internal(cps[i])){ int j=i; while(j<n&&is_han_internal(cps[j]))j++; cut_dag(g,cps,i,j,words,&wn,cap,1); i=j; }
        else { int j=i; while(j<n&&!is_han_internal(cps[j]))j++; cut_nonhan_block(cps,i,j,words,&wn,cap); i=j; }
    }
    int on=0;
    for(int wi=0; wi<wn && on<max; wi++){
        uint32_t ws=words[wi].start, we=words[wi].end; int wl=(int)(we-ws);
        if(wl>2){ for(uint32_t k=ws;k+2<=we;k++){ const jentry *e=range_lookup(g,cps,k,k+2); if(e&&e->freq!=0){ out[on].start=k;out[on].end=k+2;on++; } } }
        if(wl>3){ for(uint32_t k=ws;k+3<=we;k++){ const jentry *e=range_lookup(g,cps,k,k+3); if(e&&e->freq!=0){ out[on].start=k;out[on].end=k+3;on++; } } }
        if(on<max){ out[on].start=ws; out[on].end=we; on++; }
    }
    free(words);
    return on;
}

/* ---------------- file loading ---------------- */
static uint8_t *read_file(const char *path, size_t *len){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *b=malloc((size_t)n+1); size_t r=fread(b,1,(size_t)n,f); b[r]=0; fclose(f);
    *len=r; return b;
}
static uint32_t next_pow2(uint64_t v){ uint64_t p=1; while(p<v) p<<=1; return (uint32_t)p; }

/* append bytes to the key arena, growing as needed; returns offset */
static uint32_t kbuf_put(gsv_jieba *g, const uint8_t *src, uint32_t n){
    if((size_t)g->kbuf_len + n > g->kbuf_cap){
        size_t nc = g->kbuf_cap ? g->kbuf_cap : 4096;
        while(nc < (size_t)g->kbuf_len + n) nc *= 2;
        uint8_t *nd=realloc(g->kbuf, nc); if(!nd) return 0xFFFFFFFFu;
        g->kbuf=nd; g->kbuf_cap=nc;
    }
    memcpy(g->kbuf+g->kbuf_len, src, n);
    uint32_t off=g->kbuf_len; g->kbuf_len+=n; return off;
}

static void jinsert(gsv_jieba *g, const uint8_t *key, uint16_t klen, int64_t freq, const char *tag){
    uint32_t h=fnv_ext(FNV_OFF,key,(int)klen);
    uint32_t mask=g->ht_cap-1, idx=h&mask;
    while(g->ht[idx]) idx=(idx+1)&mask;
    jentry *e=(jentry*)g->entries + g->n_entries;
    e->koff=kbuf_put(g,key,klen); e->klen=klen; e->freq=freq;
    size_t tl=strlen(tag); if(tl>=8) tl=7; memcpy(e->tag,tag,tl); e->tag[tl]=0;
    g->ht[idx]=g->n_entries+1; g->n_entries++;
}

static int load_dict(gsv_jieba *g, const char *path){
    size_t len; uint8_t *raw=read_file(path,&len); if(!raw) return -1;
    /* pass 1 (non-mutating): count real words + total word-chars (bounds entries incl. prefixes) */
    uint64_t nreal=0, totchars=0;
    { size_t i=0;
      while(i<len){
          size_t ls=i; while(i<len && raw[i]!='\n') i++;
          size_t ll=i-ls;
          if(ll>0){
              size_t wsp=0; while(wsp<ll && raw[ls+wsp]!=' ') wsp++;
              if(wsp<ll){ int cc=0; for(size_t k=0;k<wsp;k++) if((raw[ls+k]&0xC0)!=0x80) cc++; nreal++; totchars+=(uint64_t)cc; }
          }
          i++;   /* skip '\n' (or stop at len) */
      } }
    uint64_t cap_est = nreal + totchars + 16;
    g->entries = calloc((size_t)cap_est, sizeof(jentry));
    g->kbuf = malloc(len+1); g->kbuf_cap=len+1; g->kbuf_len=0;
    g->ht_cap = next_pow2(cap_est*2);
    g->ht = calloc(g->ht_cap, sizeof(uint32_t));

    /* pass 2: insert real words (word freq tag) */
    uint64_t total_i=0;
    char *p=(char*)raw;
    while(*p){
        char *line=p; while(*p && *p!='\n') p++; if(*p=='\n')*p++=0;
        char *sp1=strchr(line,' '); if(!sp1) continue; *sp1=0;
        const char *word=line; size_t wlen=strlen(word);
        char *freqs=sp1+1; char *sp2=strchr(freqs,' ');
        int64_t freq; const char *tag="x";
        if(sp2){ *sp2=0; freq=strtoll(freqs,0,10); tag=sp2+1; }
        else { freq=strtoll(freqs,0,10); }
        jinsert(g,(const uint8_t*)word,(uint16_t)wlen,freq,tag);
        total_i += (uint64_t)(freq>0?freq:0);
    }
    /* pass 3: gen_pfdict prefix expansion - add zero-freq prefixes of each real word */
    uint32_t nreal_ins = g->n_entries;   /* first nreal entries are the real words */
    for(uint32_t i=0;i<nreal_ins;i++){
        jentry *e=(jentry*)g->entries + i;
        const uint8_t *key=g->kbuf+e->koff; uint16_t klen=e->klen;
        const char *q=(const char*)key; uint32_t c; int off=0;
        while(q < (const char*)(key+klen) && (c=gsv_utf8_decode(&q))!=0){
            off = (int)(q - (const char*)key);
            if(jlookup(g,key,(uint16_t)off)==NULL) jinsert(g,key,(uint16_t)off,0,"x");
        }
    }
    g->logtotal=log((double)total_i);
    free(raw);
    return 0;
}

typedef struct { const uint8_t *p, *end; int err; } rdcur;
static void rdu8 (rdcur *c, uint8_t *o){ if(c->p+1>c->end){c->err=1;return;} *o=*c->p++; }
static void rdu16(rdcur *c, uint16_t *o){ if(c->p+2>c->end){c->err=1;return;} memcpy(o,c->p,2); c->p+=2; }
static void rdu32(rdcur *c, uint32_t *o){ if(c->p+4>c->end){c->err=1;return;} memcpy(o,c->p,4); c->p+=4; }
static void rddbl(rdcur *c, double   *o){ if(c->p+8>c->end){c->err=1;return;} memcpy(o,c->p,8); c->p+=8; }

static int load_hmm(gsv_jieba *g, const char *path){
    size_t len; uint8_t *raw=read_file(path,&len); if(!raw) return -1;
    if(len<12 || memcmp(raw,"GSHM",4)!=0){ free(raw); return -1; }
    rdcur c; c.p=raw+4; c.end=raw+len; c.err=0;
    uint32_t ver=0, ns=0; rdu32(&c,&ver); rdu32(&c,&ns); (void)ver;
    g->n_states=ns;
    g->st_char=malloc(ns); g->st_tag=malloc(ns*4);
    for(uint32_t i=0;i<ns && !c.err;i++){
        uint8_t sc=0,tl=0,tb[3]={0}; rdu8(&c,&sc); rdu8(&c,&tl); if(c.p+3>c.end){c.err=1;break;} memcpy(tb,c.p,3); c.p+=3;
        g->st_char[i]=sc;
        for(int k=0;k<4;k++) g->st_tag[4*i+k]=(k<(int)tl)?tb[k]:0;
    }
    g->start_p=malloc(ns*sizeof(double));
    for(uint32_t i=0;i<ns && !c.err;i++) rddbl(&c,&g->start_p[i]);
    /* trans: interleaved per state [u32 count][count×(u16 to_id, f64 logp)] */
    g->tr_cnt=malloc(ns*sizeof(uint32_t)); g->tr_off=malloc(ns*sizeof(uint32_t));
    { rdcur s=c; uint64_t trtot=0;
      for(uint32_t i=0;i<ns && !s.err;i++){ uint32_t t=0; rdu32(&s,&t); s.p+=(size_t)t*10; trtot+=t; }
      g->tr_to=malloc((size_t)(trtot?trtot:1)*sizeof(uint16_t)); g->tr_lp=malloc((size_t)(trtot?trtot:1)*sizeof(double));
      uint64_t off=0;
      for(uint32_t i=0;i<ns && !c.err;i++){
          uint32_t t=0; rdu32(&c,&t); g->tr_cnt[i]=t; g->tr_off[i]=(uint32_t)off;
          for(uint32_t k=0;k<t && !c.err;k++){ rdu16(&c,&g->tr_to[off+k]); rddbl(&c,&g->tr_lp[off+k]); }
          off+=t;
      } }
    /* emit: interleaved per state [u32 count][count×(u32 cp, f64 logp)] */
    g->em_cnt=malloc(ns*sizeof(uint32_t)); g->em_off=malloc(ns*sizeof(uint32_t));
    { rdcur s=c; uint64_t emtot=0;
      for(uint32_t i=0;i<ns && !s.err;i++){ uint32_t t=0; rdu32(&s,&t); s.p+=(size_t)t*12; emtot+=t; }
      g->em_cp=malloc((size_t)(emtot?emtot:1)*sizeof(uint32_t)); g->em_lp=malloc((size_t)(emtot?emtot:1)*sizeof(double));
      uint64_t off=0;
      for(uint32_t i=0;i<ns && !c.err;i++){
          uint32_t t=0; rdu32(&c,&t); g->em_cnt[i]=t; g->em_off[i]=(uint32_t)off;
          for(uint32_t k=0;k<t && !c.err;k++){ rdu32(&c,&g->em_cp[off+k]); rddbl(&c,&g->em_lp[off+k]); }
          off+=t;
      } }
    /* char_state_tab: [u32 ncst] then per entry [u32 cp][u16 n][n×u16 ids] */
    uint32_t ncst=0; rdu32(&c,&ncst); g->n_cst=ncst;
    g->cst_cp=malloc((size_t)(ncst?ncst:1)*sizeof(uint32_t)); g->cst_cnt=malloc((size_t)(ncst?ncst:1)*sizeof(uint16_t)); g->cst_off=malloc((size_t)(ncst?ncst:1)*sizeof(uint32_t));
    { rdcur s=c; uint64_t idtot=0;
      for(uint32_t i=0;i<ncst && !s.err;i++){ uint32_t cp=0; uint16_t n=0; rdu32(&s,&cp); rdu16(&s,&n); s.p+=(size_t)n*2; idtot+=n; }
      g->cst_ids=malloc((size_t)(idtot?idtot:1)*sizeof(uint16_t));
      uint64_t off=0;
      for(uint32_t i=0;i<ncst && !c.err;i++){
          uint16_t n=0; rdu32(&c,&g->cst_cp[i]); rdu16(&c,&n); g->cst_cnt[i]=n; g->cst_off[i]=(uint32_t)off;
          for(uint16_t k=0;k<n && !c.err;k++) rdu16(&c,&g->cst_ids[off+k]);
          off+=n;
      } }
    free(raw);
    return c.err ? -1 : 0;
}

static int load_finalseg(gsv_jieba *g, const char *path){
    size_t len; uint8_t *raw=read_file(path,&len); if(!raw) return -1;
    if(len<12 || memcmp(raw,"GSFS",4)!=0){ free(raw); return -1; }
    rdcur c; c.p=raw+4; c.end=raw+len; c.err=0;
    uint32_t ver=0; rdu32(&c,&ver); (void)ver;
    for(int i=0;i<4 && !c.err;i++) rddbl(&c,&g->fs_start[i]);
    /* trans: per from-state [u32 count][count×(u8 to_id, f64 logp)] */
    { rdcur s=c; uint64_t tot=0;
      for(int i=0;i<4 && !s.err;i++){ uint32_t t=0; rdu32(&s,&t); s.p+=(size_t)t*9; tot+=t; }
      g->fs_tr_to=malloc((size_t)(tot?tot:1)); g->fs_tr_lp=malloc((size_t)(tot?tot:1)*sizeof(double));
      uint64_t off=0;
      for(int i=0;i<4 && !c.err;i++){
          uint32_t t=0; rdu32(&c,&t); g->fs_tr_cnt[i]=t; g->fs_tr_off[i]=(uint32_t)off;
          for(uint32_t k=0;k<t && !c.err;k++){ rdu8(&c,&g->fs_tr_to[off+k]); rddbl(&c,&g->fs_tr_lp[off+k]); }
          off+=t;
      } }
    /* emit: per state [u32 count][count×(u32 cp, f64 logp)] */
    { rdcur s=c; uint64_t tot=0;
      for(int i=0;i<4 && !s.err;i++){ uint32_t t=0; rdu32(&s,&t); s.p+=(size_t)t*12; tot+=t; }
      g->fs_em_cp=malloc((size_t)(tot?tot:1)*sizeof(uint32_t)); g->fs_em_lp=malloc((size_t)(tot?tot:1)*sizeof(double));
      uint64_t off=0;
      for(int i=0;i<4 && !c.err;i++){
          uint32_t t=0; rdu32(&c,&t); g->fs_em_cnt[i]=t; g->fs_em_off[i]=(uint32_t)off;
          for(uint32_t k=0;k<t && !c.err;k++){ rdu32(&c,&g->fs_em_cp[off+k]); rddbl(&c,&g->fs_em_lp[off+k]); }
          off+=t;
      } }
    free(raw);
    return c.err ? -1 : 0;
}

int gsv_jieba_load(gsv_jieba *g, const char *dir){
    memset(g,0,sizeof(*g));
    char path[1024];
    snprintf(path,sizeof path,"%s/jieba_dict.txt",dir); if(load_dict(g,path)!=0) return -1;
    snprintf(path,sizeof path,"%s/hmm.bin",dir);         if(load_hmm(g,path)!=0)  return -1;
    snprintf(path,sizeof path,"%s/finalseg.bin",dir);    if(load_finalseg(g,path)!=0) return -1;
    return 0;
}

void gsv_jieba_free(gsv_jieba *g){
    free(g->entries); free(g->kbuf); free(g->ht);
    free(g->st_char); free(g->st_tag); free(g->start_p);
    free(g->tr_cnt); free(g->tr_off); free(g->tr_to); free(g->tr_lp);
    free(g->em_cnt); free(g->em_off); free(g->em_cp); free(g->em_lp);
    free(g->cst_cp); free(g->cst_cnt); free(g->cst_off); free(g->cst_ids);
    free(g->fs_tr_to); free(g->fs_tr_lp); free(g->fs_em_cp); free(g->fs_em_lp);
    memset(g,0,sizeof(*g));
}
