#ifndef __VEC4_H__
#define __VEC4_H__

#include "common.h"
#include "vec3.h"

#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
typedef struct
{
    union
    {
        float  f[4];
        __m128 m;
    };
} mvec4;
#pragma warning(default : 4201)

MATEMATIKA_INLINE
mvec4 mate_vec4(const float x, const float y, const float z, const float w)
{
    return (mvec4){.m = _mm_setr_ps(x, y, z, w)};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_load(const float *v)
{
    return (mvec4){.m = _mm_load_ps(v)};
}

MATEMATIKA_INLINE
mvec4 mate_cvt_vec3(const vec3 vec, const float w)
{
    return (mvec4){.m = _mm_setr_ps(vec[0], vec[1], vec[2], w)};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_zero(void)
{
    return (mvec4){.m = _mm_setzero_ps()};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_set1(const float val)
{
    return (mvec4){.m = _mm_set1_ps(val)};
}

MATEMATIKA_INLINE
float mate_vec4_get(const mvec4 v, const int index)
{
    MATEMATIKA_ASSERT(index <= 3);
    return v.f[index];
}

MATEMATIKA_INLINE
mvec4 mate_vec4_add(const mvec4 v1, const mvec4 v2)
{
    return (mvec4){.m = _mm_add_ps(v1.m, v2.m)};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_sub(const mvec4 v1, const mvec4 v2)
{
    return (mvec4){.m = _mm_sub_ps(v1.m, v2.m)};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_mul(const mvec4 v1, const mvec4 v2)
{
    return (mvec4){.m = _mm_mul_ps(v1.m, v2.m)};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_scale(const mvec4 v, const float s)
{
    return (mvec4){.m = _mm_mul_ps(v.m, _mm_set1_ps(s))};
}

MATEMATIKA_INLINE
mvec4 mate_vec4_clamp(const mvec4 v, const float min, const float max)
{
    return (mvec4){.m = _mm_min_ps(_mm_max_ps(v.m, _mm_set1_ps(min)), _mm_set1_ps(max))};
}

MATEMATIKA_INLINE
mvec4 mate_cross(const mvec4 a, const mvec4 b)
{
    mvec4 shuff_a = {0};
    shuff_a.m     = _mm_shuffle_ps(a.m, a.m, _MM_SHUFFLE(3, 0, 2, 1));
    mvec4 shuff_b = {0};
    shuff_b.m     = _mm_shuffle_ps(b.m, b.m, _MM_SHUFFLE(3, 0, 2, 1));

    mvec4 sub = {0};
    sub.m     = _mm_sub_ps(
        _mm_mul_ps(a.m, shuff_b.m),
        _mm_mul_ps(b.m, shuff_a.m));

    mvec4 res = {0};
    res.m     = _mm_shuffle_ps(sub.m, sub.m, _MM_SHUFFLE(3, 0, 2, 1));
    res.f[3]  = 0.0f;

    return res;
}

MATEMATIKA_INLINE
float mate_dot(const mvec4 a, const mvec4 b)
{
#if 0
    mvec4 res = {0};
    res.m     = _mm_mul_ps(a.m, b.m);
    res.m     = _mm_hadd_ps(res.m, res.m);
    res.m     = _mm_hadd_ps(res.m, res.m);
    return res.f[0];
#elif 0
    __m128 r1   = _mm_mul_ps(a.m, b.m);
    __m128 shuf = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(r1, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#else
    return _mm_cvtss_f32(_mm_dp_ps(a.m, b.m, 0xF1));
#endif
}

MATEMATIKA_INLINE
mvec4 mate_norm(const mvec4 v)
{
    mvec4 res = {0};

    res.m = _mm_div_ps(v.m,
                       _mm_sqrt_ps(
                           _mm_set1_ps(mate_dot(v, v))));
    return res;
}

MATEMATIKA_INLINE
mvec4 mate_norm3(mvec4 v)
{
    mvec4 res = {0};

    v.f[3] = 0.0f;
    res.m  = _mm_div_ps(v.m,
                        _mm_sqrt_ps(
                           _mm_set1_ps(mate_dot(v, v))));
    return res;
}

MATEMATIKA_INLINE
mvec4 mate_reflect(const mvec4 I, const mvec4 N)
{
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
    __m128 dotProduct = _mm_dp_ps(I.m, N.m, 0x7f);                                  // Compute dot product of I and N
    __m128 scaledN    = _mm_mul_ps(N.m, _mm_mul_ps(dotProduct, _mm_set1_ps(2.0f))); // Scale N by 2 * dotProduct

    return (mvec4){.m = _mm_sub_ps(I.m, scaledN)}; // Subtract scaledN from I to get reflected vector
}

MATEMATIKA_INLINE
mvec4 mate_negate(const mvec4 m)
{
    return (mvec4){.m = _mm_mul_ps(m.m, _mm_set1_ps(-1.0f))};
}

MATEMATIKA_INLINE
mvec4 mate_surface_normal(const mvec4 A, const mvec4 B, const mvec4 C)
{
    // Compute the vectors AB and AC
    mvec4 AB = mate_vec4_sub(B, C);
    mvec4 AC = mate_vec4_sub(C, A);

    mvec4 cross = mate_cross(AB, AC);

    return mate_norm(cross);
}

#endif // __VEC4_H__