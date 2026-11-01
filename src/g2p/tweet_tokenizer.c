/* tweet_tokenizer.c - C port of NLTK TweetTokenizer (nltk.tokenize.casual).
 *
 * Faithful to TweetTokenizer().tokenize for the input alphabet produced by
 * GSV text_normalize (ASCII letters + whitespace + pause punctuation). On that
 * alphabet the only live REGEXPS alternatives are: URLS (bare-domain and
 * protocol forms), EMOTICONS, plain words [\w_]+, ellipsis dots, and the
 * single non-whitespace fallback. HANG_RE preprocessing is applied first.
 * Dead alternatives (html tags, arrows, @handles, #hashtags, emails, emoji/
 * flags, apostrophe-words, digit-numbers) require characters that
 * text_normalize strips, so omitting them is exact for this contract.
 */
#include "tweet_tokenizer.h"
#include <ctype.h>

/* ---------- byte / codepoint helpers ---------- */
static int is_sp(unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
static int is_alpha_u8(unsigned char c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z'); }
static int is_digit_u8(unsigned char c){ return c>='0'&&c<='9'; }

/* decode codepoint at s[i]; returns bytes consumed (0 if invalid/EOF). */
static int decode_at(const uint8_t *s, size_t n, size_t i, uint32_t *cp){
    if (i>=n) return 0;
    unsigned char c=s[i]; int bl;
    if (c<0x80) bl=1;
    else if ((c&0xE0)==0xC0) bl=2;
    else if ((c&0xF0)==0xE0) bl=3;
    else if ((c&0xF8)==0xF0) bl=4;
    else return 0;
    if (i+(size_t)bl>n) return 0;
    const char *p=(const char*)s+i; uint32_t d=gsv_utf8_decode(&p);
    if (!d) return 0;
    *cp=d; return (int)(p-(const char*)(s+i));
}

static int cp_is_letter(uint32_t cp){
    if (cp<0x80) return (cp>='a'&&cp<='z')||(cp>='A'&&cp<='Z');
    if (cp>=0x00C0 && cp<=0x02AF) return 1;   /* Latin ext */
    if (cp>=0x0370 && cp<=0x05FF) return 1;   /* Greek, Cyrillic */
    if (cp>=0x1E00 && cp<=0x1EFF) return 1;   /* Latin ext A/B */
    if (cp>=0x3400 && cp<=0x4DBF) return 1;   /* CJK ext A */
    if (cp>=0x4E00 && cp<=0x9FFF) return 1;   /* CJK */
    if (cp>=0xAC00 && cp<=0xD7A3) return 1;   /* Hangul */
    if (cp>=0xF900 && cp<=0xFAFF) return 1;   /* CJK compat */
    if (cp>=0xFF21 && cp<=0xFF3A) return 1;   /* fullwidth A-Z */
    if (cp>=0xFF41 && cp<=0xFF5A) return 1;   /* fullwidth a-z */
    return 0;
}
static int cp_is_number(uint32_t cp){
    if (cp<0x80) return cp>='0'&&cp<='9';
    if (cp>=0x0660 && cp<=0x0669) return 1;
    if (cp>=0xFF10 && cp<=0xFF19) return 1;
    return 0;
}
/* \w at a position (word char: letter | number | underscore). */
static int pos_is_word(const uint8_t *s, size_t n, size_t i){
    if (i>=n) return 0;
    unsigned char c=s[i];
    if (c<0x80) return is_alpha_u8(c)||is_digit_u8(c)||c=='_';
    uint32_t cp; int bl=decode_at(s,n,i,&cp); if(!bl) return 0;
    return cp_is_letter(cp)||cp_is_number(cp);
}
/* advance one codepoint from i (>=1). */
static size_t next_cp(const uint8_t *s, size_t n, size_t i){
    uint32_t cp; int bl=decode_at(s,n,i,&cp); return i+(bl?bl:1);
}

/* ---------- individual pattern matchers (return byte length, 0 = no match) ---------- */

/* plain word: [\w_]+ */
static size_t m_word(const uint8_t *s,size_t n,size_t i){
    if (i>=n || !pos_is_word(s,n,i)) return 0;
    size_t j=i; while(j<n && pos_is_word(s,n,j)) j=next_cp(s,n,j);
    return j-i;
}

/* ellipsis: \.(?:\s*\.){1,} */
static size_t m_ellipsis(const uint8_t *s,size_t n,size_t i){
    if (i>=n || s[i]!='.') return 0;
    size_t j=i+1; int groups=0;
    while (j<n){
        size_t k=j; while(k<n && is_sp(s[k])) k++;
        if (k<n && s[k]=='.'){ j=k+1; groups++; } else break;
    }
    return groups>=1 ? (j-i) : 0;
}

/* single non-whitespace: \S */
static size_t m_single(const uint8_t *s,size_t n,size_t i){
    if (i>=n || is_sp(s[i])) return 0;
    uint32_t cp; int bl=decode_at(s,n,i,&cp); return bl?bl:1;
}

/* emoticons (EMOTICONS), case-insensitive. */
static int em_eye(unsigned char c){ return c==':'||c==';'||c=='='||c=='8'; }
static int em_nose(unsigned char c){ return c=='-'||tolower(c)=='o'||c=='*'||c=='\''; } /* regex.I */
static int em_mouth(unsigned char c){
    switch(c){ case ')':case ']':case '(':case '[':case 'd':case 'D':case 'p':case 'P':
               case '/':case ':':case '}':case '{':case '@':case '|':case '\\': return 1; }
    return 0;
}
static int em_angle(unsigned char c){ return c=='<'||c=='>'; }
static size_t m_emoticon(const uint8_t *s,size_t n,size_t i){
    if (i>=n) return 0;
    unsigned char c=s[i];
    /* branch C: </?3 */
    if (c=='<'){
        size_t j=i+1;
        if (j<n && s[j]=='/'){ j++; }
        if (j<n && s[j]=='3') return j+1-i;
    }
    /* branch A: [<>]? eyes nose? mouth */
    size_t a=i;
    if (a<n && em_angle(s[a])) a++;
    if (a<n && em_eye(s[a])){
        size_t b=a+1;
        if (b<n && em_nose(s[b])) b++;
        if (b<n && em_mouth(s[b])) return b+1-i;
    }
    /* branch B: mouth nose? eyes [<>]? */
    if (em_mouth(c)){
        size_t b=i+1;
        if (b<n && em_nose(s[b])) b++;
        if (b<n && em_eye(s[b])){
            size_t d=b+1;
            if (d<n && em_angle(s[d])) d++;
            return d-i;
        }
    }
    return 0;
}

/* URLS bare-domain: (?<!@) [a-z0-9]+ (?:[.\-][a-z0-9]+){0,126} [.] [a-z]{2,13} \b /? (?!@) */
static int url_alnum(unsigned char c){ return is_alpha_u8(c)||is_digit_u8(c); }
static int url_letter(unsigned char c){ return is_alpha_u8(c); }
static int word_boundary(const uint8_t *s,size_t n,size_t e){
    int prev = (e>0) ? pos_is_word(s,n,e-1) : 0;
    int cur  = (e<n) ? pos_is_word(s,n,e)   : 0;
    return prev!=cur;
}
static size_t m_url_bare(const uint8_t *s,size_t n,size_t i){
    if (i>=n) return 0;
    if (i>0 && s[i-1]=='@') return 0;          /* (?<!@) */
    if (!url_alnum(s[i])) return 0;
    size_t p=i; while(p<n && url_alnum(s[p])) p++;   /* A = [a-z0-9]+ */
    if (p==i) return 0;
    /* greedily collect B groups: ( [.\-] [a-z0-9]+ ) up to 126 */
    size_t ends[128]; int ng=0;
    size_t q=p;
    while (ng<126 && q<n){
        if (s[q]=='.' || s[q]=='-'){
            size_t r=q+1; while(r<n && url_alnum(s[r])) r++;
            if (r>q+1){ ends[ng++]=r; q=r; continue; }
        }
        break;
    }
    /* try k = ng..0 B-groups, then [.] [a-z]{2,13} \b /? (?!@) */
    for (int k=ng;k>=0;k--){
        size_t pos = (k==0)? p : ends[k-1];
        if (pos<n && s[pos]=='.'){
            size_t t=pos+1, L=t;
            while (L<n && url_letter(s[L])) L++;
            int nl=(int)(L-t);
            if (nl>=2 && nl<=13){
                size_t e=L;                       /* greedy: all letters (<=13) */
                if (word_boundary(s,n,e)){
                    size_t f=e;
                    if (f<n && s[f]=='/') f++;
                    if (f>=n || s[f]!='@') return f-i;   /* (?!@) */
                }
            }
        }
    }
    return 0;
}

/* URLS protocol form (non-paren): https? [a-z0-9] (run of [^\s()<>{}\[\]])+ ending-class.
 * Paren-balanced alternatives are dead on the post-normalize alphabet. */
static int url_run_ok_cp(uint32_t cp){          /* [^\s()<>{}\[\]] (codepoint) */
    if (cp<0x80) return !is_sp(cp) && cp!='('&&cp!=')'&&cp!='<'&&cp!='>'&&cp!='{'&&cp!='}'&&cp!='['&&cp!=']';
    return 1;                                   /* non-ascii: not space/bracket */
}
static int url_end_ok_cp(uint32_t cp){          /* [^\s`!()\[\]{};:'".,<>?«»“”‘’] (codepoint) */
    if (cp<0x80){
        if (is_sp(cp)) return 0;
        switch(cp){ case '`':case '!':case '(':case ')':case '[':case ']':case '{':case '}':
                    case ';':case ':':case '\'':case '"':case '.':case ',':case '<':case '>':case '?':
                      return 0; }
        return 1;
    }
    switch(cp){ case 0xAB:case 0xBB:case 0x201C:case 0x201D:case 0x2018:case 0x2019: return 0; } /* «»“”‘’ */
    return 1;
}
static size_t m_url_proto(const uint8_t *s,size_t n,size_t i){
    /* match literal http / https (case-insensitive) then REQUIRED colon: https?: */
    const char *h="http";
    if (i+4>n) return 0;
    for (int k=0;k<4;k++) if (tolower(s[i+k])!=h[k]) return 0;
    size_t p=i+4;
    if (p<n && tolower(s[p])=='s') p++;
    if (p>=n || s[p]!=':') return 0;     /* https?: */
    p++;
    /* A = one [a-z0-9%] (slash form /{1,3} is dead post-normalize) */
    if (p>=n || !(url_alnum(s[p])||s[p]=='%')) return 0;
    p++;
    /* B=(run)+ then C=(ending): consume run CODEPOINTS (not bytes), need >=2,
     * last (C) url_end_ok with >=1 codepoint before it (B nonempty). */
    uint32_t cps[512]; size_t off[512]; int nc=0;
    size_t b=p;
    while (b<n && nc<512){
        uint32_t cp; int bl=decode_at(s,n,b,&cp); if(!bl) break;
        if (!url_run_ok_cp(cp)) break;
        cps[nc]=cp; off[nc]=b; nc++; b+=bl;
    }
    if (nc<2) return 0;
    for (int t=nc-1; t>0; t--) {            /* C at index t, B=[0,t) nonempty */
        if (url_end_ok_cp(cps[t])){ char tmp[4]; int e=gsv_utf8_encode(cps[t],tmp); return off[t]+e-i; }
    }
    return 0;
}

static size_t m_urls(const uint8_t *s,size_t n,size_t i){
    size_t r=m_url_proto(s,n,i); if (r) return r;
    return m_url_bare(s,n,i);
}

/* ---------- HANG_RE: ([^\p{L}\p{N}])\1{3,} -> \1\1\1 ---------- */
static int hangable(uint32_t cp){ return !cp_is_letter(cp) && !cp_is_number(cp); }
static void apply_hang(const uint8_t *s,size_t n,gsv_buf *out){
    size_t i=0;
    while (i<n){
        uint32_t cp; int bl=decode_at(s,n,i,&cp); if(!bl){ gsv_buf_append_char(out,(char)s[i]); i++; continue; }
        /* count run of identical hangable codepoints */
        size_t j=i+bl, cnt=1;
        while (j<n && cnt<64){
            uint32_t c2; int b2=decode_at(s,n,j,&c2); if(!b2) break;
            if (c2==cp && hangable(cp)){ j+=b2; cnt++; } else break;
        }
        int keep = (cnt>=4)?3:(int)cnt;
        for (int k=0;k<keep;k++){ char tmp[4]; int e=gsv_utf8_encode(cp,tmp); gsv_buf_append(out,tmp,e); }
        i=j;
    }
}

/* ---------- HTML entity fix (numeric exact + small named table; unknown removed) ---------- */
static uint32_t named_entity(const char *name,size_t len){
    struct { const char *n; uint32_t v; } T[]={
        {"amp",0x26},{"lt",0x3C},{"gt",0x3E},{"quot",0x22},{"apos",0x27},
        {"nbsp",0xA0},{"copy",0xA9},{"reg",0xAE},{"trade",0x2122},{"hellip",0x2026},
        {"mdash",0x2014},{"ndash",0x2013},{"lsquo",0x2018},{"rsquo",0x2019},
        {"ldquo",0x201C},{"rdquo",0x201D},{"middot",0xB7},{"deg",0xB0},
        {"plusmn",0xB1},{"times",0xD7},{"divide",0xF7},{"euro",0x20AC},
        {"pound",0xA3},{"cent",0xA2},{"yen",0xA5}
    };
    for (size_t k=0;k<(sizeof T/sizeof T[0]);k++)
        if (strlen(T[k].n)==len && memcmp(T[k].n,name,len)==0) return T[k].v;
    return 0xFFFFFFFFu;
}
static void cp1252(uint32_t b,uint32_t *cp){ /* map 0x80-0x9F byte -> codepoint */
    static const uint32_t M[]={0x20AC,0,0x201A,0x0192,0x201E,0x2026,0x2020,0x2021,0x02C6,0x2030,
        0x0160,0x2039,0x0152,0,0x017D,0,0,0x2018,0x2019,0x201C,0x201D,0x2022,0x2013,0x2014,
        0x02DC,0x2122,0x0161,0x203A,0x0153,0,0x017E,0x0178};
    *cp = (b>=0x80&&b<=0x9F)? M[b-0x80] : b;
}
static void replace_entities(const uint8_t *s,size_t n,gsv_buf *out){
    size_t i=0;
    while (i<n){
        if (s[i]=='&'){
            /* match &(#?(x?))([^&;\s]+); */
            size_t j=i+1; int is_num=0, is_hex=0;
            if (j<n && s[j]=='#'){ is_num=1; j++; if (j<n && (s[j]=='x'||s[j]=='X')){ is_hex=1; j++; } }
            size_t b=j;
            while (b<n && s[b]!='&' && s[b]!=';' && !is_sp(s[b])) b++;
            if (b<n && s[b]==';'){
                size_t body_len=b-j;
                uint32_t cp=0xFFFFFFFFu; int ok=0;
                if (is_num){
                    unsigned long v=0; int digits=1;
                    for (size_t k=j;k<b;k++){
                        unsigned char c=s[k];
                        if (is_hex){ v=v*16 + ((c<='9')?c-'0':(tolower(c)-'a'+10)); }
                        else { if (c<'0'||c>'9'){digits=0;break;} v=v*10+(c-'0'); }
                    }
                    if (digits && v<=0x10FFFF){
                        if (is_num && !is_hex && v>=0x80 && v<=0x9F) cp1252(v,&cp);
                        else cp=v;
                        ok=1;
                    }
                } else {
                    uint32_t nv=named_entity((const char*)s+j,body_len);
                    if (nv!=0xFFFFFFFFu){ cp=nv; ok=1; }
                }
                if (ok){ char tmp[4]; int e=gsv_utf8_encode(cp,tmp); gsv_buf_append(out,tmp,e); }
                /* else removed (remove_illegal=True) */
                i=b+1; continue;
            }
        }
        uint32_t cp; int bl=decode_at(s,n,i,&cp); if(!bl){ gsv_buf_append_char(out,(char)s[i]); i++; continue; }
        char tmp[4]; int e=gsv_utf8_encode(cp,tmp); gsv_buf_append(out,tmp,e);
        i+=bl;
    }
}

/* ---------- main tokenizer ---------- */
int gsv_tweet_tokenize(const char *text, size_t n,
                       const char ***tokens, size_t *count, void **arena){
    gsv_buf ent={0}, safe={0};
    if (gsv_buf_reserve(&ent,1)!=0) return -1;
    replace_entities((const uint8_t*)text,n,&ent);
    apply_hang((const uint8_t*)ent.data,ent.len,&safe);

    /* arena: one heap gsv_buf holding all token copies */
    gsv_buf *ar=(gsv_buf *)calloc(1,sizeof(gsv_buf));
    if (!ar){ free(ent.data); free(safe.data); return -1; }
    if (gsv_buf_reserve(ar,1)!=0){ free(ar); free(ent.data); free(safe.data); return -1; }

    const uint8_t *s=(const uint8_t*)safe.data; size_t N=safe.len;
    /* Pre-size arena so intern pointers never dangle: sum(token_len+1) <= 2N. */
    if (gsv_buf_reserve(ar, 2*N + 64) != 0){ free(ar); free(ent.data); free(safe.data); return -1; }
    size_t cap=16, tc=0;
    const char **arr=(const char **)malloc(sizeof(char*)*cap);
    if (!arr){ free(ent.data); free(safe.data); free(ar->data); return -1; }

    size_t i=0;
    while (i<N){
        size_t len=0;
        len=m_urls(s,N,i);
        if (!len) len=m_emoticon(s,N,i);
        if (!len) len=m_word(s,N,i);
        if (!len) len=m_ellipsis(s,N,i);
        if (!len) len=m_single(s,N,i);
        if (len){
            if (tc==cap){ cap*=2; const char **na=(const char **)realloc(arr,sizeof(char*)*cap);
                if(!na){ free(arr); free(ent.data); free(safe.data); free(ar->data); return -1; } arr=na; }
            char *t=gsv_buf_intern(ar,(const char*)s+i,len);
            if (!t){ free(arr); free(ent.data); free(safe.data); free(ar->data); return -1; }
            arr[tc++]=t;
            i+=len;
        } else {
            i=next_cp(s,N,i);
        }
    }
    free(ent.data); free(safe.data);
    *tokens=arr; *count=tc; *arena=ar;
    return 0;
}
