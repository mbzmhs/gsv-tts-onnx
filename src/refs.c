/* refs.c - reference (emotion preset) warmup orchestration */
#include "engine.h"
#include "features.h"
#include "g2p_dispatch.h"
#include "dsp/wav.h"
#include "dsp/resample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


static gsv_ref *find_ref(gsv_voice *v, const char *name) {
    for (int i = 0; i < v->n_refs; i++)
        if (strcmp(v->refs[i].name, name) == 0) return &v->refs[i];
    return NULL;
}

static void free_ref(gsv_ref *r) {
    free(r->prompt_codes); r->prompt_codes = NULL;
    free(r->ge); r->ge = NULL;
    free(r->ge512); r->ge512 = NULL;
    free(r->phones1); r->phones1 = NULL; r->n_phones1 = 0;
    free(r->bert); r->bert = NULL;
}

static int gsv_warmup_ref_impl(gsv_engine *e, const char *ref_name,
                    const char *wav_path, const char *text, const char *lang) {
    if (!e || !e->has_voice) return GSV_ERR_NOT_LOADED;
    if (!ref_name || !ref_name[0] || !wav_path || !wav_path[0]) {
        gsv_set_error(e, "gsv_warmup_ref: bad args");
        return GSV_ERR_INVALID_ARG;
    }

    /* load + downmix to mono float32 */
    gsv_wav wav;
    if (gsv_wav_load(wav_path, &wav) != 0) {
        gsv_set_error(e, "gsv_warmup_ref: cannot read wav %s", wav_path);
        return GSV_ERR_IO;
    }
    int sr0 = wav.sample_rate;

    /* --- ge path: resample -> 32k, peak-normalize, STFT + fbank(16k) --- */
    gsv_resampler r32, r16b, r16a;
    if (gsv_resampler_create(&r32, sr0, 32000) != 0) { gsv_wav_free(&wav); return GSV_ERR_NO_MEMORY; }
    int64_t n32 = 0;
    float *wav32 = gsv_resample(&r32, wav.data, wav.n, &n32);
    gsv_resampler_destroy(&r32);
    if (!wav32 || n32 <= 0) { gsv_wav_free(&wav); return GSV_ERR_GENERIC; }

    float maxx = 0.f;
    for (int64_t i = 0; i < n32; i++) { float a = fabsf(wav32[i]); if (a > maxx) maxx = a; }
    if (maxx > 1.f) {
        float div = maxx < 2.f ? maxx : 2.f;
        for (int64_t i = 0; i < n32; i++) wav32[i] /= div;
    }

    if (gsv_resampler_create(&r16b, 32000, 16000) != 0) { free(wav32); gsv_wav_free(&wav); return GSV_ERR_NO_MEMORY; }
    int64_t n16b = 0;
    float *wav16b = gsv_resample(&r16b, wav32, n32, &n16b);
    gsv_resampler_destroy(&r16b);
    if (!wav16b || n16b <= 0) { free(wav32); gsv_wav_free(&wav); return GSV_ERR_GENERIC; }

    /* slot (replace existing ref with same name) */
    gsv_ref *ref = find_ref(&e->voice, ref_name);
    if (!ref) {
        e->voice.refs = (gsv_ref *)realloc(e->voice.refs,
            sizeof(gsv_ref) * (size_t)(e->voice.n_refs + 1));
        if (!e->voice.refs) { free(wav32); free(wav16b); gsv_wav_free(&wav); return GSV_ERR_NO_MEMORY; }
        ref = &e->voice.refs[e->voice.n_refs++];
        memset(ref, 0, sizeof(*ref));
    } else {
        free_ref(ref);
    }
    snprintf(ref->name, sizeof(ref->name), "%s", ref_name);
    snprintf(ref->wav_path, sizeof(ref->wav_path), "%s", wav_path);
    if (text) snprintf(ref->text, sizeof(ref->text), "%s", text);
    if (lang) snprintf(ref->lang, sizeof(ref->lang), "%s", lang);

    int rc = gsv_ref_compute_ge(e, &e->voice.cfg, wav32, n32, wav16b, n16b, ref);
    free(wav32); free(wav16b);
    if (rc != GSV_OK) { gsv_wav_free(&wav); return rc; }

    /* --- prompt codes path: direct sr -> 16k + 0.3s trailing silence --- */
    if (gsv_resampler_create(&r16a, sr0, 16000) != 0) { gsv_wav_free(&wav); return GSV_ERR_NO_MEMORY; }
    int64_t n16a = 0;
    float *wav16a = gsv_resample(&r16a, wav.data, wav.n, &n16a);
    gsv_resampler_destroy(&r16a);
    gsv_wav_free(&wav);
    if (!wav16a || n16a <= 0) return GSV_ERR_GENERIC;

    int64_t silence_n = 16000 / 10 * 3;   /* 0.3s */
    float *wav16c = (float *)malloc(sizeof(float) * (size_t)(n16a + silence_n));
    if (!wav16c) { free(wav16a); return GSV_ERR_NO_MEMORY; }
    memcpy(wav16c, wav16a, sizeof(float) * (size_t)n16a);
    memset(wav16c + n16a, 0, sizeof(float) * (size_t)silence_n);
    free(wav16a);

    rc = gsv_ref_compute_codes(e, wav16c, n16a + silence_n, ref);
    free(wav16c);
    if (rc != GSV_OK) return rc;

    /* phones1 + bert1 (M6/M6b): G2P of ref text -> prompt phoneme ids, and (zh)
     * per-phone BERT features. Best-effort: if the dispatcher or this language is
     * unavailable, phones1/bert1 stay empty (BERT degrades to zeros). */
    if (ref->text[0] && e->g2p) {
        const char *rlang = ref->lang[0] ? ref->lang : "zh";
        gsv_g2p_full_result fr;
        if (gsv_g2p_dispatch_full(e->g2p, ref->text, rlang, &fr) == GSV_OK && fr.n_ids > 0) {
            free(ref->phones1);
            ref->phones1 = fr.ids;      /* take ownership of the ids array */
            ref->n_phones1 = fr.n_ids;
            fr.ids = NULL;              /* so result_free won't double-free it */

            free(ref->bert); ref->bert = NULL;
            if (!strcmp(rlang, "zh") && e->bert_tok.n && fr.n_words == fr.n_phs && fr.n_words > 0) {
                float *bf = (float *)malloc(sizeof(float) * (size_t)fr.n_ids * 1024);
                if (bf && gsv_bert_phone_features(e, fr.words, fr.n_words, fr.phs, fr.n_ids, bf) == GSV_OK) {
                    ref->bert = bf;
                } else {
                    free(bf);   /* alignment/backend failure -> zeros fallback */
                }
            }
        }
        gsv_g2p_full_result_free(&fr);   /* frees words/phs (ids already taken) */
    }

    return GSV_OK;
}

int gsv_warmup_ref(gsv_engine *e, const char *ref_name,
                    const char *wav_path, const char *text, const char *lang) {
    if (!e) return GSV_ERR_NOT_LOADED;
    pthread_mutex_lock(&e->infer_lock);
    int rc = gsv_warmup_ref_impl(e, ref_name, wav_path, text, lang);
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

int gsv_warmup_refs(gsv_engine *e,
                     const char *const *ref_names, const char *const *wav_paths,
                     const char *const *texts, const char *const *langs, int n) {
    if (!e || n <= 0 || !ref_names || !wav_paths) return GSV_ERR_INVALID_ARG;
    pthread_mutex_lock(&e->infer_lock);
    int rc = GSV_OK;
    for (int i = 0; i < n; i++) {
        rc = gsv_warmup_ref_impl(e, ref_names[i], wav_paths[i],
                                  texts ? texts[i] : NULL, langs ? langs[i] : NULL);
        if (rc != GSV_OK) break;
    }
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}

static int gsv_unload_ref_impl(gsv_engine *e, const char *ref_name) {
    if (!e || !e->has_voice) return GSV_ERR_NOT_LOADED;
    gsv_voice *v = &e->voice;
    for (int i = 0; i < v->n_refs; i++) {
        if (strcmp(v->refs[i].name, ref_name) == 0) {
            free_ref(&v->refs[i]);
            memmove(&v->refs[i], &v->refs[i + 1],
                    sizeof(gsv_ref) * (size_t)(v->n_refs - i - 1));
            v->n_refs--;
            return GSV_OK;
        }
    }
    gsv_set_error(e, "gsv_unload_ref: ref '%s' not found", ref_name);
    return GSV_ERR_REF_MISSING;
}

int gsv_unload_ref(gsv_engine *e, const char *ref_name) {
    if (!e) return GSV_ERR_NOT_LOADED;
    pthread_mutex_lock(&e->infer_lock);
    int rc = gsv_unload_ref_impl(e, ref_name);
    pthread_mutex_unlock(&e->infer_lock);
    return rc;
}
