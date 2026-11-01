/* json.c - minimal recursive-descent JSON parser */
#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef struct { const char *p; const char *end; int err; } jp;

static void skip_ws(jp *s) {
    while (s->p < s->end) {
        char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') s->p++;
        else break;
    }
}

static jv *parse_value(jp *s);

static jv *new_jv(jv_type t) {
    jv *v = (jv *)calloc(1, sizeof(jv));
    if (v) v->type = t;
    return v;
}

void jv_free(jv *v) {
    if (!v) return;
    if (v->type == JV_STR && v->str) free(v->str);
    if ((v->type == JV_ARR || v->type == JV_OBJ) && v->items) {
        for (int i = 0; i < v->len; i++) jv_free(v->items[i]);
        free(v->items);
    }
    free(v);
}

static char *parse_string_raw(jp *s) {
    if (s->p >= s->end || *s->p != '"') { s->err = 1; return NULL; }
    s->p++;
    size_t cap = 16, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { s->err = 1; return NULL; }
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\' && s->p < s->end) {
            char e = *s->p++;
            switch (e) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    if (s->p + 4 > s->end) { s->err = 1; free(buf); return NULL; }
                    unsigned cp = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = *s->p++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= h - '0';
                        else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                        else { s->err = 1; free(buf); return NULL; }
                    }
                    /* encode as UTF-8 (BMP only; surrogate pairs -> replacement) */
                    if (len + 4 >= cap) { cap *= 2; char *nb = realloc(buf, cap); if (!nb) { s->err=1; free(buf); return NULL; } buf = nb; }
                    if (cp < 0x80) buf[len++] = (char)cp;
                    else if (cp < 0x800) {
                        buf[len++] = (char)(0xC0 | (cp >> 6));
                        buf[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        buf[len++] = (char)(0xE0 | (cp >> 12));
                        buf[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        buf[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    continue;
                }
                default: s->err = 1; free(buf); return NULL;
            }
        }
        if (len + 2 >= cap) { cap *= 2; char *nb = realloc(buf, cap); if (!nb) { s->err=1; free(buf); return NULL; } buf = nb; }
        buf[len++] = c;
    }
    if (s->p >= s->end) { s->err = 1; free(buf); return NULL; }
    s->p++; /* closing quote */
    buf[len] = 0;
    char *out = malloc(len + 1);
    memcpy(out, buf, len + 1);
    free(buf);
    return out;
}

static jv *parse_object(jp *s) {
    s->p++; /* { */
    jv *v = new_jv(JV_OBJ);
    if (!v) { s->err = 1; return NULL; }
    int cap = 8, n = 0;
    v->items = (jv **)calloc(cap, sizeof(jv *));
    skip_ws(s);
    if (s->p < s->end && *s->p == '}') { s->p++; return v; }
    while (s->p < s->end) {
        skip_ws(s);
        char *key = parse_string_raw(s);
        if (s->err || !key) { jv_free(v); return NULL; }
        skip_ws(s);
        if (s->p >= s->end || *s->p != ':') { free(key); jv_free(v); return NULL; }
        s->p++;
        jv *val = parse_value(s);
        if (!val) { free(key); jv_free(v); return NULL; }
        if (n + 2 > cap) { cap *= 2; v->items = realloc(v->items, cap * sizeof(jv *)); }
        /* store key as JV_STR item */
        jv *kv = new_jv(JV_STR); kv->str = key;
        v->items[n++] = kv;
        v->items[n++] = val;
        skip_ws(s);
        if (s->p < s->end && *s->p == ',') { s->p++; continue; }
        if (s->p < s->end && *s->p == '}') { s->p++; break; }
        jv_free(v); return NULL;
    }
    v->len = n / 2;
    return v;
}

static jv *parse_array(jp *s) {
    s->p++; /* [ */
    jv *v = new_jv(JV_ARR);
    if (!v) { s->err = 1; return NULL; }
    int cap = 8, n = 0;
    v->items = (jv **)calloc(cap, sizeof(jv *));
    skip_ws(s);
    if (s->p < s->end && *s->p == ']') { s->p++; return v; }
    while (s->p < s->end) {
        jv *val = parse_value(s);
        if (!val) { jv_free(v); return NULL; }
        if (n + 1 > cap) { cap *= 2; v->items = realloc(v->items, cap * sizeof(jv *)); }
        v->items[n++] = val;
        skip_ws(s);
        if (s->p < s->end && *s->p == ',') { s->p++; continue; }
        if (s->p < s->end && *s->p == ']') { s->p++; break; }
        jv_free(v); return NULL;
    }
    v->len = n;
    return v;
}

static jv *parse_value(jp *s) {
    skip_ws(s);
    if (s->p >= s->end) { s->err = 1; return NULL; }
    char c = *s->p;
    if (c == '{') return parse_object(s);
    if (c == '[') return parse_array(s);
    if (c == '"') {
        char *str = parse_string_raw(s);
        if (s->err || !str) return NULL;
        jv *v = new_jv(JV_STR); if (!v) { free(str); return NULL; }
        v->str = str; return v;
    }
    if (c == 't' && s->end - s->p >= 4 && !memcmp(s->p, "true", 4)) {
        s->p += 4; jv *v = new_jv(JV_BOOL); if (v) v->boolean = 1; return v;
    }
    if (c == 'f' && s->end - s->p >= 5 && !memcmp(s->p, "false", 5)) {
        s->p += 5; jv *v = new_jv(JV_BOOL); if (v) v->boolean = 0; return v;
    }
    if (c == 'n' && s->end - s->p >= 4 && !memcmp(s->p, "null", 4)) {
        s->p += 4; return new_jv(JV_NULL);
    }
    /* number */
    char *numend;
    double d = strtod(s->p, &numend);
    if (numend == s->p) { s->err = 1; return NULL; }
    s->p = numend;
    jv *v = new_jv(JV_NUM);
    if (v) v->num = d;
    return v;
}

jv *jv_parse(const char *text) {
    if (!text) return NULL;
    jp s = { text, text + strlen(text), 0 };
    jv *v = parse_value(&s);
    if (s.err) { jv_free(v); return NULL; }
    skip_ws(&s);
    return v;
}

const jv *jv_obj_get(const jv *obj, const char *key) {
    if (!obj || obj->type != JV_OBJ) return NULL;
    for (int i = 0; i < obj->len; i++) {
        const jv *k = obj->items[2 * i];
        if (k && k->type == JV_STR && k->str && !strcmp(k->str, key))
            return obj->items[2 * i + 1];
    }
    return NULL;
}

const jv *jv_arr_get(const jv *arr, int i) {
    if (!arr || arr->type != JV_ARR || i < 0 || i >= arr->len) return NULL;
    return arr->items[i];
}
