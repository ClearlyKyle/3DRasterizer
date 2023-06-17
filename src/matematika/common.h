#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>

// MATEMATIKA

#if defined(_MSC_VER)
#define MATEMATIKA_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define MATEMATIKA_INLINE static inline __attribute((always_inline))
#else
#error "Unsupported compiler"
#endif

#define MATEMATIKA_ASSERT(EXPR) assert(EXPR)

#define M_PI          3.14159265358979323846264338327950288
#define MATE_D2R(DEG) ((DEG)*M_PI / 180.0)
#define MATE_R2D(RAD) ((RAD)*180.0 / M_PI)

#define MATE_D2RF(DEG) (float)MATE_D2R(DEG)
#define MATE_R2DF(RAD) (float)MATE_R2D(RAD)

#endif // __COMMON_H__