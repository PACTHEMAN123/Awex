/* Read-only call counters for native AState qualification via LD_PRELOAD. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stddef.h>

static _Atomic unsigned long long registrations;
static _Atomic unsigned long long verbs_registrations;

unsigned long long astate_probe_reg_calls(void) { return atomic_load(&registrations); }
unsigned long long astate_probe_mr_calls(void) { return atomic_load(&verbs_registrations); }

static void *resolve(const char *name) {
    void *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        void *library = dlopen("libutrans.so", RTLD_LAZY | RTLD_NOLOAD);
        if (library) symbol = dlsym(library, name);
    }
    if (!symbol) {
        fprintf(stderr, "AState observer cannot resolve %s: %s\n", name, dlerror());
        abort();
    }
    return symbol;
}

const void *utrans_regist_ram(void *context, void *address, size_t length, int numa) {
    typedef const void *(*function)(void *, void *, size_t, int);
    const void *result = ((function)resolve("utrans_regist_ram"))(context, address, length, numa);
    if (result) atomic_fetch_add(&registrations, 1);
    return result;
}

void *ibv_reg_mr(void *pd, void *address, size_t length, int access) {
    typedef void *(*function)(void *, void *, size_t, int);
    void *result = ((function)resolve("ibv_reg_mr"))(pd, address, length, access);
    if (result) atomic_fetch_add(&verbs_registrations, 1);
    return result;
}

void *ibv_reg_mr_iova2(void *pd, void *address, size_t length, uint64_t iova, unsigned int access) {
    typedef void *(*function)(void *, void *, size_t, uint64_t, unsigned int);
    void *result = ((function)resolve("ibv_reg_mr_iova2"))(pd, address, length, iova, access);
    if (result) atomic_fetch_add(&verbs_registrations, 1);
    return result;
}
