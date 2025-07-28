
#ifndef DL_SYM_H
#define DL_SYM_H

#ifdef __cplusplus
extern "C" {
#endif

#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        return (void*) f; }                                      \

void* __dlsym_hook_section(void* handle, const char* symbol);

#ifdef __cplusplus
}
#endif

#endif
