/* tone_sandhi.c - pure-C port of ToneSandhi (GPT_SoVITS/G2P/Chinese/tone_sandhi.py).
 * Replicates pre_merge_for_modify and modified_tone byte-for-byte. */
#include "tone_sandhi.h"
#include "ts_sets.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* special chars (UTF-8) */
#define C_BU "\xe4\xb8\x8d"   /* 不 U+4E0D */
#define C_YI "\xe4\xb8\x80"   /* 一 U+4E00 */
#define C_ER "\xe5\x84\xbf"   /* 儿 U+513F */

/* char-class sets (UTF-8 literal strings) */
static const char *SET_PARTICLE = "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶";
static const char *SET_DE       = "的地得";
static const char *SET_LZG      = "了着过";
static const char *SET_MENZI    = "们子";
static const char *SET_SXL      = "上下里";
static const char *SET_LAIQU    = "来去";
static const char *SET_PREV_LQ  = "上下进出回过起开";
static const char *SET_MEAS     = "几有两半多各整每做是";
static const char *PUNC         = "：，；。？！“”‘’':,;.?!";

/* ---------------- small helpers ---------------- */
static uint32_t *to_cps(const char *s, int *n){
    int cap=16,cnt=0; uint32_t *a=malloc(cap*4); const char*p=s; uint32_t c;
    while((c=gsv_utf8_decode(&p))!=0){ if(cnt==cap){cap*=2;a=realloc(a,cap*4);} a[cnt++]=c; }
    *n=cnt; return a;
}
static int cp_in_set(uint32_t cp, const char *set){
    const char*p=set; uint32_t c;
    while((c=gsv_utf8_decode(&p))!=0) if(c==cp) return 1;
    return 0;
}
/* match Python str.isnumeric() for the chars that can precede 个 in Chinese text:
 * ASCII digits + CJK numeral ideographs (category Nl). */
static int is_numeric(uint32_t cp){
    if(cp>='0'&&cp<='9') return 1;
    switch(cp){
        case 0x4E00: /*一*/ case 0x4E8C: /*二*/ case 0x4E09: /*三*/ case 0x56DB: /*四*/
        case 0x4E94: /*五*/ case 0x516D: /*六*/ case 0x4E03: /*七*/ case 0x516B: /*八*/
        case 0x4E5D: /*九*/ case 0x5341: /*十*/ case 0x767E: /*百*/ case 0x5343: /*千*/
        case 0x4E07: /*万*/ case 0x4EBF: /*亿*/ case 0x96F6: /*零*/ case 0x3007: /*〇*/
        case 0x5EFF: /*廿*/ case 0x5345: /*卅*/ case 0x4EE8: /*仨*/ return 1;
    }
    return 0;
}
static void set_tone(char *f, int t){ size_t L=strlen(f); f[L-1]=(char)('0'+t); }

/* is cps[a..b) (as UTF-8) present in the given NULL-terminated set array? */
static int range_in_set(const uint32_t *cps, int a, int b, const char **set){
    if(b<=a) return 0;
    char tmp[128]; int t=0; for(int k=a;k<b;k++) t+=gsv_utf8_encode(cps[k],tmp+t); tmp[t]=0;
    for(int i=0;set[i];i++) if(strcmp(tmp,set[i])==0) return 1;
    return 0;
}
static int last2_in_set(const uint32_t *cps, int a, int b, const char **set){
    int L=b-a; if(L<=0) return 0; int s=(L>=2)?(b-2):a; return range_in_set(cps,s,b,set);
}

static int all_tone3(char **f, int n){ for(int i=0;i<n;i++) if(f[i][strlen(f[i])-1]!='3') return 0; return 1; }
static int all_tone3_range(char **f, int a, int b){ for(int i=a;i<b;i++) if(f[i][strlen(f[i])-1]!='3') return 0; return 1; }

static int find_cp(const uint32_t *cps, int n, uint32_t t){ for(int i=0;i<n;i++) if(cps[i]==t) return i; return -1; }
static int is_redup_str(const char *w){
    const char*p=w; uint32_t a=gsv_utf8_decode(&p), b=gsv_utf8_decode(&p);
    return a!=0 && b!=0 && *p==0 && a==b;
}

/* arena join of 1-3 parts (b/c may be NULL) into a gsv_buf, returns NUL-terminated ptr.
 * Args are copied to temps first so that a realloc inside the reserve cannot dangle them
 * (they may point into `ar` itself). */
static const char *ar_join(gsv_buf *ar, const char *a, const char *b, const char *c){
    size_t la=strlen(a), lb=b?strlen(b):0, lc=c?strlen(c):0;
    char *ta=malloc(la+1); memcpy(ta,a,la); ta[la]=0;
    char *tb=NULL,*tc=NULL;
    if(b){ tb=malloc(lb+1); memcpy(tb,b,lb); tb[lb]=0; }
    if(c){ tc=malloc(lc+1); memcpy(tc,c,lc); tc[lc]=0; }
    if(gsv_buf_reserve(ar, la+lb+lc+1)!=0){ free(ta);free(tb);free(tc); return NULL; }
    char *p=ar->data+ar->len;
    memcpy(p,ta,la);
    if(b)memcpy(p+la,tb,lb);
    if(c)memcpy(p+la+lb,tc,lc);
    p[la+lb+lc]=0; ar->len+=la+lb+lc+1;
    free(ta);free(tb);free(tc);
    return p;
}

/* ---------------- _split_word ---------------- */
static void split_word(gsv_jieba *g, const uint32_t *cps, int n, gsv_range out[2]){
    gsv_range *rs=malloc(sizeof(gsv_range)*(2*n+8));
    int rn=gsv_cut_for_search(g, cps, n, rs, 2*n+8);
    int Lmin=1<<30, fs=0, fe=0;
    for(int i=0;i<rn;i++){ int l=(int)(rs[i].end-rs[i].start); if(l<Lmin){ Lmin=l; fs=rs[i].start; fe=rs[i].end; } }
    (void)fe;
    int idx=-1;
    for(int p=0;p+Lmin<=n;p++){ int ok=1; for(int k=0;k<Lmin;k++) if(cps[p+k]!=cps[fs+k]){ok=0;break;} if(ok){idx=p;break;} }
    if(idx==0){ out[0].start=0; out[0].end=Lmin; out[1].start=Lmin; out[1].end=n; }
    else      { out[0].start=0; out[0].end=n-Lmin; out[1].start=n-Lmin; out[1].end=n; }
    free(rs);
}

/* ---------------- modified_tone sub-rules ---------------- */
static void bu_sandhi(const uint32_t *cps, int n, char **finals){
    if(n==3 && cps[1]==0x4E0D){ set_tone(finals[1],5); }
    else { for(int i=0;i<n;i++) if(cps[i]==0x4E0D && i+1<n && finals[i+1][strlen(finals[i+1])-1]=='4') set_tone(finals[i],2); }
}

static void yi_sandhi(const uint32_t *cps, int n, char **finals){
    if(find_cp(cps,n,0x4E00)>=0){
        int allnum=1; for(int i=0;i<n;i++) if(cps[i]!=0x4E00 && !is_numeric(cps[i])){allnum=0;break;}
        if(allnum) return;
    }
    if(n==3 && cps[1]==0x4E00 && cps[0]==cps[2]) set_tone(finals[1],5);
    else if(n>=2 && cps[0]==0x7B2C && cps[1]==0x4E00) set_tone(finals[1],1);   /* 第一 */
    else {
        for(int i=0;i<n;i++)
            if(cps[i]==0x4E00 && i+1<n){
                if(finals[i+1][strlen(finals[i+1])-1]=='4') set_tone(finals[i],2);
                else if(!cp_in_set(cps[i+1], PUNC)) set_tone(finals[i],4);
            }
    }
}

static void neural_sandhi(gsv_jieba *g, const uint32_t *cps, int n, const char *pos, char **finals){
    int in_mustnot = range_in_set(cps,0,n,TS_MUST_NOT_NEURAL);
    for(int j=1;j<n;j++)
        if(cps[j]==cps[j-1] && (pos[0]=='n'||pos[0]=='v'||pos[0]=='a') && !in_mustnot) set_tone(finals[j],5);

    int ge_idx=find_cp(cps,n,0x4E2A);   /* 个 */
    uint32_t last=cps[n-1];
    if(n>=1 && cp_in_set(last,SET_PARTICLE)) set_tone(finals[n-1],5);
    else if(n>=1 && cp_in_set(last,SET_DE)) set_tone(finals[n-1],5);
    else if(n==1 && cp_in_set(cps[0],SET_LZG) && (strcmp(pos,"ul")==0||strcmp(pos,"uz")==0||strcmp(pos,"ug")==0)) set_tone(finals[n-1],5);
    else if(n>1 && cp_in_set(last,SET_MENZI) && (strcmp(pos,"r")==0||strcmp(pos,"n")==0) && !in_mustnot) set_tone(finals[n-1],5);
    else if(n>1 && cp_in_set(last,SET_SXL) && (pos[0]=='s'||pos[0]=='l'||pos[0]=='f')) set_tone(finals[n-1],5);
    else if(n>1 && cp_in_set(last,SET_LAIQU) && cp_in_set(cps[n-2],SET_PREV_LQ)) set_tone(finals[n-1],5);
    else if((ge_idx>=1 && (is_numeric(cps[ge_idx-1]) || cp_in_set(cps[ge_idx-1],SET_MEAS))) || (n==1 && cps[0]==0x4E2A)) set_tone(finals[ge_idx],5);
    else {
        if(range_in_set(cps,0,n,TS_MUST_NEURAL) || last2_in_set(cps,0,n,TS_MUST_NEURAL)) set_tone(finals[n-1],5);
    }

    gsv_range sp[2]; split_word(g,cps,n,sp);
    int a0=sp[0].start, a1=sp[0].end;
    if(range_in_set(cps,a0,a1,TS_MUST_NEURAL) || last2_in_set(cps,a0,a1,TS_MUST_NEURAL)){ if(a1-1>=a0) set_tone(finals[a1-1],5); }
    if(range_in_set(cps,a1,n ,TS_MUST_NEURAL) || last2_in_set(cps,a1,n ,TS_MUST_NEURAL)){ if(n-1>=a1)   set_tone(finals[n-1],5); }
}

static void three_sandhi(gsv_jieba *g, const uint32_t *cps, int n, char **finals){
    if(n==2 && all_tone3(finals,n)) set_tone(finals[0],2);
    else if(n==3){
        gsv_range sp[2]; split_word(g,cps,n,sp);
        int a0=sp[0].start, a1=sp[0].end; int L0=a1-a0, L1=n-a1;
        if(all_tone3(finals,n)){
            if(L0==2){ set_tone(finals[0],2); set_tone(finals[1],2); }
            else if(L0==1){ set_tone(finals[1],2); }
        } else {
            if(L0==2 && all_tone3_range(finals,a0,a1)) set_tone(finals[a0],2);
            if(L1==2 && all_tone3_range(finals,a1,n))  set_tone(finals[a1],2);
            else if(!all_tone3_range(finals,a1,n) && finals[a1][strlen(finals[a1])-1]=='3' && finals[a1-1][strlen(finals[a1-1])-1]=='3') set_tone(finals[a1-1],2);
        }
    }
    else if(n==4){
        if(all_tone3_range(finals,0,2)) set_tone(finals[0],2);
        if(all_tone3_range(finals,2,4)) set_tone(finals[2],2);
    }
}

void gsv_ts_modified_tone(gsv_jieba *g, const char *word, const char *pos, char **finals, int n){
    uint32_t *cps=to_cps(word,&n);
    bu_sandhi(cps,n,finals);
    yi_sandhi(cps,n,finals);
    neural_sandhi(g,cps,n,pos,finals);
    three_sandhi(g,cps,n,finals);
    free(cps);
}

/* ---------------- pre_merge_for_modify sub-steps ---------------- */
static void seg_push(gsv_ts_seg *out,int *on,int max,const char*w,const char*p){ if(*on<max){out[*on].w=w;out[*on].p=p;(*on)++;} }

static int merge_bu(gsv_buf*ar,const gsv_ts_seg*in,int n,gsv_ts_seg*out,int max){
    int on=0; const char*last="";
    for(int i=0;i<n;i++){
        const char*w=in[i].w;
        if(strcmp(last,C_BU)==0) w=ar_join(ar,last,in[i].w,NULL);
        if(strcmp(w,C_BU)!=0) seg_push(out,&on,max,w,in[i].p);
        last=w;
    }
    if(strcmp(last,C_BU)==0) seg_push(out,&on,max,last,"d");
    return on;
}

static int merge_yi(gsv_buf*ar,const gsv_ts_seg*in,int n,gsv_ts_seg*out,int max){
    gsv_ts_seg*t1=malloc(sizeof(gsv_ts_seg)*max);
    int on=0, i=0;
    while(i<n){
        const char*w=in[i].w,*p=in[i].p; int merged=0;
        if(i-1>=0 && strcmp(w,C_YI)==0 && i+1<n){
            gsv_ts_seg last=(on>0)?t1[on-1]:in[i-1];
            if(strcmp(last.w,in[i+1].w)==0 && strcmp(last.p,"v")==0 && strcmp(in[i+1].p,"v")==0){
                t1[on-1].w=ar_join(ar,last.w,C_YI,in[i+1].w);
                t1[on-1].p=last.p; i+=2; merged=1;
            }
        }
        if(!merged){ seg_push(t1,&on,max,w,p); i++; }
    }
    int on2=0;
    for(int k=0;k<on;k++){
        if(on2>0 && strcmp(out[on2-1].w,C_YI)==0) out[on2-1].w=ar_join(ar,out[on2-1].w,t1[k].w,NULL);
        else seg_push(out,&on2,max,t1[k].w,t1[k].p);
    }
    free(t1);
    return on2;
}

static int merge_redup(gsv_buf*ar,const gsv_ts_seg*in,int n,gsv_ts_seg*out,int max){
    int on=0;
    for(int i=0;i<n;i++)
        if(on>0 && strcmp(in[i].w,out[on-1].w)==0) out[on-1].w=ar_join(ar,out[on-1].w,in[i].w,NULL);
        else seg_push(out,&on,max,in[i].w,in[i].p);
    return on;
}

static int merge_cont3(gsv_g2p_zh*py,gsv_buf*ar,const gsv_ts_seg*in,int n,gsv_ts_seg*out,int max){
    int*at3=malloc(n*sizeof(int));
    for(int i=0;i<n;i++){
        int L=gsv_utf8_len(in[i].w); char**fs=malloc((L?L:1)*sizeof(char*)); int fn=0;
        gsv_pinyin_lazy(py,in[i].w,GSV_PY_STYLE_FINALS3,fs,&fn);
        at3[i]=(fn>0)?all_tone3(fs,fn):1; free(fs);
    }
    int*ml=malloc(n*sizeof(int)); memset(ml,0,n*sizeof(int));
    int on=0;
    for(int i=0;i<n;i++){
        if(i-1>=0 && at3[i-1] && at3[i] && !ml[i-1]){
            int lp=gsv_utf8_len(in[i-1].w), lc=gsv_utf8_len(in[i].w);
            if(!is_redup_str(in[i-1].w) && (lp+lc)<=3){ out[on-1].w=ar_join(ar,out[on-1].w,in[i].w,NULL); ml[i]=1; }
            else seg_push(out,&on,max,in[i].w,in[i].p);
        } else seg_push(out,&on,max,in[i].w,in[i].p);
    }
    free(at3); free(ml); return on;
}

static int merge_cont3_2(gsv_g2p_zh*py,gsv_buf*ar,const gsv_ts_seg*in,int n,gsv_ts_seg*out,int max){
    char*first=malloc(n),*last=malloc(n); memset(first,0,n); memset(last,0,n);
    for(int i=0;i<n;i++){
        int L=gsv_utf8_len(in[i].w); char**fs=malloc((L?L:1)*sizeof(char*)); int fn=0;
        gsv_pinyin_lazy(py,in[i].w,GSV_PY_STYLE_FINALS3,fs,&fn);
        if(fn>0){ first[i]=fs[0][strlen(fs[0])-1]; last[i]=fs[fn-1][strlen(fs[fn-1])-1]; }
        free(fs);
    }
    int*ml=malloc(n*sizeof(int)); memset(ml,0,n*sizeof(int));
    int on=0;
    for(int i=0;i<n;i++){
        if(i-1>=0 && last[i-1]=='3' && first[i]=='3' && !ml[i-1]){
            int lp=gsv_utf8_len(in[i-1].w), lc=gsv_utf8_len(in[i].w);
            if(!is_redup_str(in[i-1].w) && (lp+lc)<=3){ out[on-1].w=ar_join(ar,out[on-1].w,in[i].w,NULL); ml[i]=1; }
            else seg_push(out,&on,max,in[i].w,in[i].p);
        } else seg_push(out,&on,max,in[i].w,in[i].p);
    }
    free(first); free(last); free(ml); return on;
}

static int merge_er(gsv_buf*ar,const gsv_ts_seg*in,int n,gsv_ts_seg*out,int max){
    int on=0;
    for(int i=0;i<n;i++)
        if(i-1>=0 && strcmp(in[i].w,C_ER)==0 && strcmp(in[i-1].w,"#")!=0) out[on-1].w=ar_join(ar,out[on-1].w,in[i].w,NULL);
        else seg_push(out,&on,max,in[i].w,in[i].p);
    return on;
}

int gsv_ts_pre_merge(gsv_jieba *g, gsv_g2p_zh *py, gsv_buf *ar, const gsv_ts_seg *in, int n, gsv_ts_seg *out, int max){
    (void)g;
    /* Pre-size the arena so it never reallocs during this call: every merge step writes
     * at most n strings each <= total input bytes, across 6 steps. A realloc would dangle
     * pointers already stored in b1/b2/out. */
    uint64_t total=0; for(int i=0;i<n;i++) total+=strlen(in[i].w);
    /* worst case (e.g. merge_er on a run of 儿) writes ~n/2 * total per heavy step;
     * use a generous multiple so the arena never reallocs (a realloc would dangle the
     * pointers already stored in b1/b2/out). */
    size_t bound=(size_t)total*(size_t)n*16 + 8192; if(bound<8192) bound=8192;
    gsv_buf_reserve(ar, bound);
    gsv_ts_seg*b1=malloc(sizeof(gsv_ts_seg)*max),*b2=malloc(sizeof(gsv_ts_seg)*max);
    int nn;
    nn=merge_bu(ar,in,n,b1,max);
    nn=merge_yi(ar,b1,nn,b2,max);
    nn=merge_redup(ar,b2,nn,b1,max);
    nn=merge_cont3(py,ar,b1,nn,b2,max);
    nn=merge_cont3_2(py,ar,b2,nn,b1,max);
    nn=merge_er(ar,b1,nn,out,max);
    free(b1); free(b2);
    return nn;
}
