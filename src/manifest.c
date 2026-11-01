/* manifest.c - parse manifest.json into typed config */
#include "manifest.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_file(const char *path, char **out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return -1; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[rd] = 0;
    *out = buf;
    return 0;
}

static int get_int(const jv *obj, const char *key, int def) {
    const jv *v = jv_obj_get(obj, key);
    if (v && v->type == JV_NUM) return (int)v->num;
    return def;
}

static void get_str(const jv *obj, const char *key, char *dst, size_t n) {
    dst[0] = 0;
    const jv *v = jv_obj_get(obj, key);
    if (v && v->type == JV_STR) {
        strncpy(dst, v->str, n - 1);
        dst[n - 1] = 0;
    }
}

static void get_names(const jv *obj, const char *key, char (*dst)[64], int *n) {
    *n = 0;
    const jv *arr = jv_obj_get(obj, key);
    if (!arr || arr->type != JV_ARR) return;
    for (int i = 0; i < arr->len && i < 8; i++) {
        const jv *s = jv_arr_get(arr, i);
        if (s && s->type == JV_STR) {
            strncpy(dst[i], s->str, 63);
            dst[i][63] = 0;
            (*n)++;
        }
    }
}

static void parse_graph(const jv *obj, gsv_graph_cfg *g) {
    memset(g, 0, sizeof(*g));
    get_str(obj, "onnx", g->onnx, sizeof(g->onnx));
    get_str(obj, "vocab", g->vocab, sizeof(g->vocab));   /* bert only */
    get_names(obj, "input_names", g->input_names, &g->n_in);
    get_names(obj, "output_names", g->output_names, &g->n_out);
    g->dim = get_int(obj, "dim", 0);
    g->gin_channels = get_int(obj, "gin_channels", 0);
    const jv *dt = jv_obj_get(obj, "dtype");
    g->fp16 = (dt && dt->type == JV_STR && strcmp(dt->str, "fp16") == 0) ? 1 : 0;
}

int gsv_manifest_load(const char *path, gsv_manifest *m) {
    memset(m, 0, sizeof(*m));
    char *text = NULL;
    if (read_file(path, &text) != 0) return -1;
    jv *root = jv_parse(text);
    free(text);
    if (!root || root->type != JV_OBJ) { if (root) jv_free(root); return -1; }

    m->contract = get_int(root, "contract", 1);
    get_str(root, "tts_version", m->tts_version, sizeof(m->tts_version));
    m->sample_rate = get_int(root, "sample_rate", 32000);
    m->semantic_hz = get_int(root, "semantic_hz", 25);

    const jv *o;

    /* per-character model locations: prefix + fallback dir (no voices list) */
    o = jv_obj_get(root, "gpt");
    if (o && o->type == JV_OBJ) {
        get_str(o, "prefix", m->gpt_prefix, sizeof(m->gpt_prefix));
        get_str(o, "fallback", m->gpt_fallback, sizeof(m->gpt_fallback));
    }
    o = jv_obj_get(root, "sovits");
    if (o && o->type == JV_OBJ) {
        get_str(o, "prefix", m->sovits_prefix, sizeof(m->sovits_prefix));
        get_str(o, "fallback", m->sovits_fallback, sizeof(m->sovits_fallback));
        gsv_voice_cfg *c = &m->base_voice;
        c->inter_channels = get_int(o, "inter_channels", 0);
        c->hidden_channels = get_int(o, "hidden_channels", 0);
        c->filter_channels = get_int(o, "filter_channels", 0);
        c->n_heads = get_int(o, "n_heads", 0);
        c->n_layers = get_int(o, "n_layers", 0);
        c->kernel_size = get_int(o, "kernel_size", 3);
        c->gin_channels = get_int(o, "gin_channels", 0);
        c->samples_per_frame = get_int(o, "samples_per_frame", 320);
        const jv *ea = jv_obj_get(o, "enc_p");
        if (ea && ea->type == JV_OBJ) {
            c->ly_max = get_int(ea, "ly_max", 1024);
            c->lt_max = get_int(ea, "lt_max", 512);
        } else { c->ly_max = 1024; c->lt_max = 512; }
        const jv *sp = jv_obj_get(o, "spec");
        if (sp && sp->type == JV_OBJ) {
            c->spec_filter_length = get_int(sp, "filter_length", 2048);
            c->spec_hop_length = get_int(sp, "hop_length", 320);
            c->spec_win_length = get_int(sp, "win_length", 1280);
        }
    }

    o = jv_obj_get(root, "t2s");
    if (o && o->type == JV_OBJ) {
        m->t2s.max_len = get_int(o, "max_len", 1024);
        m->t2s.num_layers = get_int(o, "num_layers", 24);
        m->t2s.num_heads = get_int(o, "num_heads", 16);
        m->t2s.model_dim = get_int(o, "model_dim", 512);
        m->t2s.phoneme_vocab_size = get_int(o, "phoneme_vocab_size", 732);
        m->t2s.vocab_size = get_int(o, "vocab_size", 1025);
        m->t2s.eos = get_int(o, "eos", 1024);
        const jv *sup = jv_obj_get(o, "suppressed");
        if (sup && sup->type == JV_ARR) {
            m->t2s.n_suppressed = sup->len;
            m->t2s.suppressed = (int *)calloc((size_t)(sup->len ? sup->len : 1), sizeof(int));
            for (int i = 0; i < sup->len; i++) {
                const jv *n = jv_arr_get(sup, i);
                if (n && n->type == JV_NUM) m->t2s.suppressed[i] = (int)n->num;
            }
        }
    }

    o = jv_obj_get(root, "bert");   if (o) parse_graph(o, &m->bert);
    o = jv_obj_get(root, "hubert"); if (o) parse_graph(o, &m->hubert);
    o = jv_obj_get(root, "sv");     if (o) parse_graph(o, &m->sv);
    o = jv_obj_get(root, "ge");     if (o) parse_graph(o, &m->ge);

    o = jv_obj_get(root, "vq");
    if (o && o->type == JV_OBJ) {
        get_str(o, "codebook", m->vq.codebook, sizeof(m->vq.codebook));
        m->vq.dim = get_int(o, "dim", 768);
        m->vq.bins = get_int(o, "bins", 1024);
        m->vq.n_q = get_int(o, "n_q", 1);
        const jv *sp = jv_obj_get(o, "ssl_proj");
        if (sp && sp->type == JV_OBJ) {
            m->vq.ssl_proj_kernel = get_int(sp, "kernel", 2);
            m->vq.ssl_proj_stride = get_int(sp, "stride", 2);
            get_str(sp, "file", m->vq.ssl_proj_file, sizeof(m->vq.ssl_proj_file));
        } else { m->vq.ssl_proj_kernel = 2; m->vq.ssl_proj_stride = 2; }
    }

    jv_free(root);
    return 0;
}

void gsv_manifest_free(gsv_manifest *m) {
    free(m->t2s.suppressed);
    m->t2s.suppressed = NULL;
}
