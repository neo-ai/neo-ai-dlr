/*
 * Copyright 2020 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License 2.0 (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#include <openssl/core_numbers.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/types.h>
#include <openssl/params.h>
#include "prov/bio.h"
#include "prov/implementations.h"
#include "serializer_local.h"

static OSSL_OP_serializer_newctx_fn x25519_pub_newctx;
static OSSL_OP_serializer_newctx_fn x448_pub_newctx;
static OSSL_OP_serializer_freectx_fn ecx_pub_freectx;
static OSSL_OP_serializer_serialize_data_fn ecx_pub_der_data;
static OSSL_OP_serializer_serialize_object_fn ecx_pub_der;
static OSSL_OP_serializer_serialize_data_fn ecx_pub_pem_data;
static OSSL_OP_serializer_serialize_object_fn ecx_pub_pem;

static OSSL_OP_serializer_serialize_data_fn ecx_pub_print_data;
static OSSL_OP_serializer_serialize_object_fn ecx_pub_print;

/*
 * Context used for public key serialization.
 */
struct ecx_pub_ctx_st {
    void *provctx;
    ECX_KEY_TYPE type;
};

/* Public key : context */
static void *ecx_pub_newctx(void *provctx, ECX_KEY_TYPE type)
{
    struct ecx_pub_ctx_st *ctx = OPENSSL_zalloc(sizeof(*ctx));

    if (ctx != NULL) {
        ctx->provctx = provctx;
        ctx->type = type;
    }
    return ctx;
}

static void *x25519_pub_newctx(void *provctx)
{
    return ecx_pub_newctx(provctx, ECX_KEY_TYPE_X25519);
}

static void *x448_pub_newctx(void *provctx)
{
    return ecx_pub_newctx(provctx, ECX_KEY_TYPE_X448);
}

static void ecx_pub_freectx(void *ctx)
{
    OPENSSL_free(ctx);
}

/* Public key : DER */
static int ecx_pub_der_data(void *vctx, const OSSL_PARAM params[], BIO *out,
                            OSSL_PASSPHRASE_CALLBACK *cb, void *cbarg)
{
    struct ecx_pub_ctx_st *ctx = vctx;
    OSSL_OP_keymgmt_new_fn *ecx_new;
    OSSL_OP_keymgmt_free_fn *ecx_free;
    OSSL_OP_keymgmt_import_fn *ecx_import;
    int ok = 0;

    ecx_get_new_free_import(ctx->type, &ecx_new, &ecx_free, &ecx_import);

    if (ecx_import != NULL) {
        ECX_KEY *ecxkey;

        if ((ecxkey = ecx_new(ctx->provctx)) != NULL
            && ecx_import(ecxkey, OSSL_KEYMGMT_SELECT_KEYPAIR, params)
            && ecx_pub_der(ctx, ecxkey, out, cb, cbarg))
            ok = 1;
        ecx_free(ecxkey);
    }
    return ok;
}

static int ecx_pub_der(void *vctx, void *ecxkey, BIO *out,
                       OSSL_PASSPHRASE_CALLBACK *cb, void *cbarg)
{
    struct ecx_pub_ctx_st *ctx = vctx;

    return ossl_prov_write_pub_der_from_obj(out, ecxkey,
                                            ctx->type == ECX_KEY_TYPE_X25519
                                            ? EVP_PKEY_X25519 : EVP_PKEY_X448,
                                            NULL,
                                            ossl_prov_ecx_pub_to_der);
}

/* Public key : PEM */
static int ecx_pub_pem_data(void *vctx, const OSSL_PARAM params[], BIO *out,
                            OSSL_PASSPHRASE_CALLBACK *cb, void *cbarg)
{
    struct ecx_pub_ctx_st *ctx = vctx;
    OSSL_OP_keymgmt_new_fn *ecx_new;
    OSSL_OP_keymgmt_free_fn *ecx_free;
    OSSL_OP_keymgmt_import_fn *ecx_import;
    int ok = 0;

    ecx_get_new_free_import(ctx->type, &ecx_new, &ecx_free, &ecx_import);

    if (ecx_import != NULL) {
        ECX_KEY *ecxkey;

        if ((ecxkey = ecx_new(ctx->provctx)) != NULL
            && ecx_import(ecxkey, OSSL_KEYMGMT_SELECT_KEYPAIR, params)
            && ecx_pub_pem(ctx, ecxkey, out, cb, cbarg))
            ok = 1;
        ecx_free(ecxkey);
    }
    return ok;
}

static int ecx_pub_pem(void *vctx, void *ecxkey, BIO *out,
                       OSSL_PASSPHRASE_CALLBACK *cb, void *cbarg)
{
    struct ecx_pub_ctx_st *ctx = vctx;

    return ossl_prov_write_pub_pem_from_obj(out, ecxkey,
                                            ctx->type == ECX_KEY_TYPE_X25519
                                            ? EVP_PKEY_X25519 : EVP_PKEY_X448,
                                            NULL,
                                            ossl_prov_ecx_pub_to_der);

}

static int ecx_pub_print_data(void *vctx, const OSSL_PARAM params[], BIO *out,
                              OSSL_PASSPHRASE_CALLBACK *cb, void *cbarg)
{
    struct ecx_pub_ctx_st *ctx = vctx;
    OSSL_OP_keymgmt_new_fn *ecx_new;
    OSSL_OP_keymgmt_free_fn *ecx_free;
    OSSL_OP_keymgmt_import_fn *ecx_import;
    int ok = 0;

    ecx_get_new_free_import(ctx->type, &ecx_new, &ecx_free, &ecx_import);

    if (ecx_import != NULL) {
        ECX_KEY *ecxkey;

        if ((ecxkey = ecx_new(ctx)) != NULL
            && ecx_import(ecxkey, OSSL_KEYMGMT_SELECT_KEYPAIR, params)
            && ecx_pub_print(ctx, ecxkey, out, cb, cbarg))
            ok = 1;
        ecx_free(ecxkey);
    }
    return ok;
}

static int ecx_pub_print(void *ctx, void *ecxkey, BIO *out,
                         OSSL_PASSPHRASE_CALLBACK *cb, void *cbarg)
{
    return ossl_prov_print_ecx(out, ecxkey, ecx_print_pub);
}

#define MAKE_SERIALIZER_FUNCTIONS(alg, type) \
    const OSSL_DISPATCH alg##_pub_##type##_serializer_functions[] = { \
        { OSSL_FUNC_SERIALIZER_NEWCTX, (void (*)(void))alg##_pub_newctx }, \
        { OSSL_FUNC_SERIALIZER_FREECTX, (void (*)(void))ecx_pub_freectx }, \
        { OSSL_FUNC_SERIALIZER_SERIALIZE_DATA, \
          (void (*)(void))ecx_pub_##type##_data }, \
        { OSSL_FUNC_SERIALIZER_SERIALIZE_OBJECT, \
          (void (*)(void))ecx_pub_##type }, \
        { 0, NULL } \
    };

#define MAKE_SERIALIZER_FUNCTIONS_GROUP(alg) \
    MAKE_SERIALIZER_FUNCTIONS(alg, der) \
    MAKE_SERIALIZER_FUNCTIONS(alg, pem) \
    MAKE_SERIALIZER_FUNCTIONS(alg, print)

MAKE_SERIALIZER_FUNCTIONS_GROUP(x25519)
MAKE_SERIALIZER_FUNCTIONS_GROUP(x448)
