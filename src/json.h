/* json.h - minimal JSON parser (objects/arrays/strings/numbers/bool/null) */
#ifndef GSV_JSON_H
#define GSV_JSON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    JV_NULL, JV_BOOL, JV_NUM, JV_STR, JV_ARR, JV_OBJ
} jv_type;

typedef struct jv jv;

struct jv {
    jv_type type;
    /* NUM */ double num;
    /* BOOL */ int boolean;
    /* STR */ char *str;        /* malloc'd, NUL-terminated */
    /* ARR/OBJ */ jv **items;   /* ARR: elements; OBJ: key,value,key,value... */
    int len;                    /* ARR: count; OBJ: number of keys */
};

/* Parse a JSON document. Returns root value or NULL on error.
 * All memory owned by the tree; release with jv_free(). */
jv *jv_parse(const char *text);
void jv_free(jv *v);

/* Lookup helper for objects: returns item value or NULL. */
const jv *jv_obj_get(const jv *obj, const char *key);
/* Array index (0-based), bounds-checked. */
const jv *jv_arr_get(const jv *arr, int i);

#ifdef __cplusplus
}
#endif

#endif
