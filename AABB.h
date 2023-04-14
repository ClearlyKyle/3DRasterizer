#ifndef __AABB_H__
#define __AABB_H__

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

union AABB_u
{
    struct
    {
        int maxX;
        int minX;
        int maxY;
        int minY;
    };
    int values[4];
};

static __m128i Get_AABB_SIMD(const __m128 v1, const __m128 v2, const __m128 v3, int screen_width, int screen_height)
{
    const __m128i vec1 = _mm_cvtps_epi32(v1);
    const __m128i vec2 = _mm_cvtps_epi32(v2);
    const __m128i vec3 = _mm_cvtps_epi32(v3);

    __m128i min_values = _mm_and_si128(_mm_max_epi32(_mm_min_epi32(_mm_min_epi32(vec1, vec2), vec3), _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFE));
    __m128i max_values = _mm_min_epi32(_mm_add_epi32(_mm_max_epi32(_mm_max_epi32(vec1, vec2), vec3), _mm_set1_epi32(1)), _mm_set_epi32(0, 0, screen_height - 1, screen_width - 1));

    // Returns {maxX, minX, maxY, minY}
    return _mm_unpacklo_epi32(max_values, min_values);
}

#endif // __AABB_H__