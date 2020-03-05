/*
 * Copyright 2019 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License 2.0 (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#include <openssl/core_names.h>
#include "internal/cryptlib.h"
#include "internal/nelem.h"
#include "crypto/evp.h"
#include "crypto/asn1.h"
#include "internal/core.h"
#include "internal/provider.h"
#include "evp_local.h"

/*
 * match_type() checks if two EVP_KEYMGMT are matching key types.  This
 * function assumes that the caller has made all the necessary NULL checks.
 */
static int match_type(const EVP_KEYMGMT *keymgmt1, const EVP_KEYMGMT *keymgmt2)
{
    const OSSL_PROVIDER *prov2 = EVP_KEYMGMT_provider(keymgmt2);
    const char *name2 = evp_first_name(prov2, EVP_KEYMGMT_number(keymgmt2));

    return EVP_KEYMGMT_is_a(keymgmt1, name2);
}

struct import_data_st {
    EVP_KEYMGMT *keymgmt;
    void *keydata;

    int selection;
};

static int try_import(const OSSL_PARAM params[], void *arg)
{
    struct import_data_st *data = arg;

    return evp_keymgmt_import(data->keymgmt, data->keydata, data->selection,
                              params);
}

void *evp_keymgmt_util_export_to_provider(EVP_PKEY *pk, EVP_KEYMGMT *keymgmt)
{
    void *keydata = NULL;
    struct import_data_st import_data;
    size_t i = 0;

    /* Export to where? */
    if (keymgmt == NULL)
        return NULL;

    /* If we have an unassigned key, give up */
    if (pk->keymgmt == NULL)
        return NULL;

    /* If |keymgmt| matches the "origin" |keymgmt|, no more to do */
    if (pk->keymgmt == keymgmt)
        return pk->keydata;

    /* If this key is already exported to |keymgmt|, no more to do */
    i = evp_keymgmt_util_find_operation_cache_index(pk, keymgmt);
    if (i < OSSL_NELEM(pk->operation_cache)
        && pk->operation_cache[i].keymgmt != NULL)
        return pk->operation_cache[i].keydata;

    /* If the "origin" |keymgmt| doesn't support exporting, give up */
    /*
     * TODO(3.0) consider an evp_keymgmt_export() return value that indicates
     * that the method is unsupported.
     */
    if (pk->keymgmt->export == NULL)
        return NULL;

    /* Check that we have found an empty slot in the export cache */
    /*
     * TODO(3.0) Right now, we assume we have ample space.  We will have to
     * think about a cache aging scheme, though, if |i| indexes outside the
     * array.
     */
    if (!ossl_assert(i < OSSL_NELEM(pk->operation_cache)))
        return NULL;

    /*
     * Make sure that the type of the keymgmt to export to matches the type
     * of the "origin"
     */
    if (!ossl_assert(match_type(pk->keymgmt, keymgmt)))
        return NULL;

    /* Create space to import data into */
    if ((keydata = evp_keymgmt_newdata(keymgmt)) == NULL)
        return NULL;

    /*
     * We look at the already cached provider keys, and import from the
     * first that supports it (i.e. use its export function), and export
     * the imported data to the new provider.
     */

    /* Setup for the export callback */
    import_data.keydata = keydata;
    import_data.keymgmt = keymgmt;
    import_data.selection = OSSL_KEYMGMT_SELECT_ALL;

    /*
     * The export function calls the callback (try_import), which does the
     * import for us.  If successful, we're done.
     */
    if (!evp_keymgmt_export(pk->keymgmt, pk->keydata, OSSL_KEYMGMT_SELECT_ALL,
                            &try_import, &import_data)) {
        /* If there was an error, bail out */
        evp_keymgmt_freedata(keymgmt, keydata);
        return NULL;
    }

    /* Add the new export to the operation cache */
    if (!evp_keymgmt_util_cache_keydata(pk, i, keymgmt, keydata)) {
        evp_keymgmt_freedata(keymgmt, keydata);
        return NULL;
    }

    return keydata;
}

void evp_keymgmt_util_clear_operation_cache(EVP_PKEY *pk)
{
    size_t i, end = OSSL_NELEM(pk->operation_cache);

    if (pk != NULL) {
        for (i = 0; i < end && pk->operation_cache[i].keymgmt != NULL; i++) {
            EVP_KEYMGMT *keymgmt = pk->operation_cache[i].keymgmt;
            void *keydata = pk->operation_cache[i].keydata;

            pk->operation_cache[i].keymgmt = NULL;
            pk->operation_cache[i].keydata = NULL;
            evp_keymgmt_freedata(keymgmt, keydata);
            EVP_KEYMGMT_free(keymgmt);
        }
    }
}

size_t evp_keymgmt_util_find_operation_cache_index(EVP_PKEY *pk,
                                                   EVP_KEYMGMT *keymgmt)
{
    size_t i, end = OSSL_NELEM(pk->operation_cache);

    for (i = 0; i < end && pk->operation_cache[i].keymgmt != NULL; i++) {
        if (keymgmt == pk->operation_cache[i].keymgmt)
            break;
    }

    return i;
}

int evp_keymgmt_util_cache_keydata(EVP_PKEY *pk, size_t index,
                                   EVP_KEYMGMT *keymgmt, void *keydata)
{
    if (keydata != NULL) {
        if (!EVP_KEYMGMT_up_ref(keymgmt))
            return 0;
        pk->operation_cache[index].keydata = keydata;
        pk->operation_cache[index].keymgmt = keymgmt;
    }
    return 1;
}

void evp_keymgmt_util_cache_keyinfo(EVP_PKEY *pk)
{
    /*
     * Cache information about the provider "origin" key.
     *
     * This services functions like EVP_PKEY_size, EVP_PKEY_bits, etc
     */
    if (pk->keymgmt != NULL) {
        int bits = 0;
        int security_bits = 0;
        int size = 0;
        OSSL_PARAM params[4];

        params[0] = OSSL_PARAM_construct_int(OSSL_PKEY_PARAM_BITS, &bits);
        params[1] = OSSL_PARAM_construct_int(OSSL_PKEY_PARAM_SECURITY_BITS,
                                             &security_bits);
        params[2] = OSSL_PARAM_construct_int(OSSL_PKEY_PARAM_MAX_SIZE, &size);
        params[3] = OSSL_PARAM_construct_end();
        if (evp_keymgmt_get_params(pk->keymgmt, pk->keydata, params)) {
            pk->cache.size = size;
            pk->cache.bits = bits;
            pk->cache.security_bits = security_bits;
        }
    }
}

void *evp_keymgmt_util_fromdata(EVP_PKEY *target, EVP_KEYMGMT *keymgmt,
                                int selection, const OSSL_PARAM params[])
{
    void *keydata = evp_keymgmt_newdata(keymgmt);

    if (keydata != NULL) {
        if (!evp_keymgmt_import(keymgmt, keydata, selection, params)
            || !EVP_KEYMGMT_up_ref(keymgmt)) {
            evp_keymgmt_freedata(keymgmt, keydata);
            return NULL;
        }

        evp_keymgmt_util_clear_operation_cache(target);
        target->keymgmt = keymgmt;
        target->keydata = keydata;
        evp_keymgmt_util_cache_keyinfo(target);
    }

    return keydata;
}

int evp_keymgmt_util_has(EVP_PKEY *pk, int selection)
{
    /* Check if key is even assigned */
    if (pk->keymgmt == NULL)
        return 0;

    return evp_keymgmt_has(pk->keymgmt, pk->keydata, selection);
}

/*
 * evp_keymgmt_util_match() doesn't just look at the provider side "origin",
 * but also in the operation cache to see if there's any common keymgmt that
 * supplies OP_keymgmt_match.
 *
 * evp_keymgmt_util_match() adheres to the return values that EVP_PKEY_cmp()
 * and EVP_PKEY_cmp_parameters() return, i.e.:
 *
 *  1   same key
 *  0   not same key
 * -1   not same key type
 * -2   unsupported operation
 */
int evp_keymgmt_util_match(EVP_PKEY *pk1, EVP_PKEY *pk2, int selection)
{
    EVP_KEYMGMT *keymgmt1 = NULL, *keymgmt2 = NULL;
    void *keydata1 = NULL, *keydata2 = NULL;

    if (pk1 == NULL || pk2 == NULL) {
        if (pk1 == NULL && pk2 == NULL)
            return 1;
        return 0;
    }

    keymgmt1 = pk1->keymgmt;
    keydata1 = pk1->keydata;
    keymgmt2 = pk2->keymgmt;
    keydata2 = pk2->keydata;

    if (keymgmt1 != keymgmt2) {
        void *tmp_keydata = NULL;

        /* Complex case, where the keymgmt differ */
        if (keymgmt1 != NULL
            && keymgmt2 != NULL
            && !match_type(keymgmt1, keymgmt2)) {
            ERR_raise(ERR_LIB_EVP, EVP_R_DIFFERENT_KEY_TYPES);
            return -1;           /* Not the same type */
        }

        /*
         * The key types are determined to match, so we try cross export,
         * but only to keymgmt's that supply a matching function.
         */
        if (keymgmt2 != NULL
            && keymgmt2->match != NULL) {
            tmp_keydata = evp_keymgmt_util_export_to_provider(pk1, keymgmt2);
            if (tmp_keydata != NULL) {
                keymgmt1 = keymgmt2;
                keydata1 = tmp_keydata;
            }
        }
        if (tmp_keydata == NULL
            && keymgmt1 != NULL
            && keymgmt1->match != NULL) {
            tmp_keydata = evp_keymgmt_util_export_to_provider(pk2, keymgmt1);
            if (tmp_keydata != NULL) {
                keymgmt2 = keymgmt1;
                keydata2 = tmp_keydata;
            }
        }
    }

    /* If we still don't have matching keymgmt implementations, we give up */
    if (keymgmt1 != keymgmt2)
        return -2;

    return evp_keymgmt_match(keymgmt1, keydata1, keydata2, selection);
}

int evp_keymgmt_util_copy(EVP_PKEY *to, EVP_PKEY *from, int selection)
{
    /* Save copies of pointers we want to play with without affecting |to| */
    EVP_KEYMGMT *to_keymgmt = to->keymgmt;
    void *to_keydata = to->keydata, *alloc_keydata = NULL;

    /* An unassigned key can't be copied */
    if (from == NULL || from->keymgmt == NULL)
        return 0;

    /* If |from| doesn't support copying, we fail */
    if (from->keymgmt->copy == NULL)
        return 0;

    /* If |to| doesn't have a provider side "origin" yet, create one */
    if (to_keymgmt == NULL) {
        to_keydata = alloc_keydata = evp_keymgmt_newdata(from->keymgmt);
        if (to_keydata == NULL)
            return 0;
        to_keymgmt = from->keymgmt;
    }

    if (to_keymgmt == from->keymgmt) {
        /* |to| and |from| have the same keymgmt, just copy and be done */
        if (!evp_keymgmt_copy(to_keymgmt, to_keydata, from->keydata,
                              selection))
            return 0;
    } else if (match_type(to_keymgmt, from->keymgmt)) {
        struct import_data_st import_data;

        import_data.keymgmt = to_keymgmt;
        import_data.keydata = to_keydata;
        import_data.selection = selection;

        if (!evp_keymgmt_export(from->keymgmt, from->keydata, selection,
                                &try_import, &import_data)) {
            evp_keymgmt_freedata(to_keymgmt, alloc_keydata);
            return 0;
        }
    } else {
        ERR_raise(ERR_LIB_EVP, EVP_R_DIFFERENT_KEY_TYPES);
        return 0;
    }

    if (to->keymgmt == NULL
        && !EVP_KEYMGMT_up_ref(to_keymgmt)) {
        evp_keymgmt_freedata(to_keymgmt, alloc_keydata);
        return 0;
    }
    evp_keymgmt_util_clear_operation_cache(to);
    to->keymgmt = to_keymgmt;
    to->keydata = to_keydata;
    evp_keymgmt_util_cache_keyinfo(to);

    return 1;
}
