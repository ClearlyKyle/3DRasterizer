#ifndef __UTILS_H__
#define __UTILS_H__

#include <assert.h>

#define ASSERT(EXPR)      assert(EXPR)
#define UTILS_UNUSED(VAR) ((void)(VAR))

#define ASSERTF(EXPR, MESSAGE, ...)                \
    if (!(EXPR))                                   \
    {                                              \
        fprintf(stderr, (MESSAGE), ##__VA_ARGS__); \
        ASSERT(EXPR);                              \
    }

#endif // __UTILS_H__