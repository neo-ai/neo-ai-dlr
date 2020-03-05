/*
 * Copyright 2019 The OpenSSL Project Authors. All Rights Reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates.  All rights reserved.
 *
 * Licensed under the Apache License 2.0 (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#ifndef OSSL_INTERNAL_PROPERTY_H
# define OSSL_INTERNAL_PROPERTY_H

#include "internal/cryptlib.h"

typedef struct ossl_method_store_st OSSL_METHOD_STORE;
typedef struct ossl_property_list_st OSSL_PROPERTY_LIST;

/* Initialisation */
int ossl_property_parse_init(OPENSSL_CTX *ctx);

/* Property definition parser */
OSSL_PROPERTY_LIST *ossl_parse_property(OPENSSL_CTX *ctx, const char *defn);
/* Property query parser */
OSSL_PROPERTY_LIST *ossl_parse_query(OPENSSL_CTX *ctx, const char *s);
/* Property checker of query vs definition */
int ossl_property_match_count(const OSSL_PROPERTY_LIST *query,
                              const OSSL_PROPERTY_LIST *defn);
/* Free a parsed property list */
void ossl_property_free(OSSL_PROPERTY_LIST *p);


/* Implementation store functions */
OSSL_METHOD_STORE *ossl_method_store_new(OPENSSL_CTX *ctx);
void ossl_method_store_free(OSSL_METHOD_STORE *store);
int ossl_method_store_add(OSSL_METHOD_STORE *store, const OSSL_PROVIDER *prov,
                          int nid, const char *properties, void *method,
                          int (*method_up_ref)(void *),
                          void (*method_destruct)(void *));
int ossl_method_store_remove(OSSL_METHOD_STORE *store, int nid,
                             const void *method);
int ossl_method_store_fetch(OSSL_METHOD_STORE *store, int nid,
                            const char *prop_query, void **result);
int ossl_method_store_set_global_properties(OSSL_METHOD_STORE *store,
                                            const char *prop_query);

/* property query cache functions */
int ossl_method_store_cache_get(OSSL_METHOD_STORE *store, int nid,
                                const char *prop_query, void **result);
int ossl_method_store_cache_set(OSSL_METHOD_STORE *store, int nid,
                                const char *prop_query, void *result,
                                int (*method_up_ref)(void *),
                                void (*method_destruct)(void *));
#endif
