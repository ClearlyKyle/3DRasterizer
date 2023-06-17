#ifndef __MAT4_H__
#define __MAT4_H__

#include "common.h"
#include "vec4.h"
#include "vec3.h"

#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
typedef struct
{
    union
    {
        float  f[4][4]; // Do we actually need access to this?
        __m128 m[4];
    };
} mmat4;
#pragma warning(default : 4201)

MATEMATIKA_INLINE
mmat4 mate_mat_add(const mmat4 a, const mmat4 b)
{
    mmat4 res = {0};
    res.m[0]  = _mm_add_ps(a.m[0], b.m[0]);
    res.m[1]  = _mm_add_ps(a.m[1], b.m[1]);
    res.m[2]  = _mm_add_ps(a.m[2], b.m[2]);
    res.m[3]  = _mm_add_ps(a.m[3], b.m[3]);
    return res;
}

MATEMATIKA_INLINE
mvec4 mate_mat_mulv(const mmat4 m, const mvec4 v)
{
    const __m128 v0 = _mm_shuffle_ps(v.m, v.m, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 v1 = _mm_shuffle_ps(v.m, v.m, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 v2 = _mm_shuffle_ps(v.m, v.m, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 v3 = _mm_shuffle_ps(v.m, v.m, _MM_SHUFFLE(3, 3, 3, 3));

    mvec4 res = {0};
    res.m     = _mm_mul_ps(m.m[3], v3);
    res.m     = _mm_add_ps(res.m, _mm_mul_ps(m.m[2], v2));
    res.m     = _mm_add_ps(res.m, _mm_mul_ps(m.m[1], v1));
    res.m     = _mm_add_ps(res.m, _mm_mul_ps(m.m[0], v0));
    return res;
}

MATEMATIKA_INLINE
mmat4 mate_mat_mul(const mmat4 m1, const mmat4 m2)
{
#if 1
    mmat4 res = {0};
    for (int i = 0; i < 4; i++)
    {
        const __m128 brod1 = _mm_shuffle_ps(m2.m[i], m2.m[i], _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 brod2 = _mm_shuffle_ps(m2.m[i], m2.m[i], _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 brod3 = _mm_shuffle_ps(m2.m[i], m2.m[i], _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 brod4 = _mm_shuffle_ps(m2.m[i], m2.m[i], _MM_SHUFFLE(3, 3, 3, 3));
        const __m128 row   = _mm_add_ps(_mm_add_ps(
                                          _mm_mul_ps(brod1, m1.m[0]),
                                          _mm_mul_ps(brod2, m1.m[1])),
                                        _mm_add_ps(
                                          _mm_mul_ps(brod3, m1.m[2]),
                                          _mm_mul_ps(brod4, m1.m[3])));
        res.m[i]           = row;
    }
    return res;
#else
    mmat4 res = {0};

    __m128 brod1 = _mm_shuffle_ps(m2.m[0], m2.m[0], _MM_SHUFFLE(0, 0, 0, 0));
    __m128 brod2 = _mm_shuffle_ps(m2.m[0], m2.m[0], _MM_SHUFFLE(1, 1, 1, 1));
    __m128 brod3 = _mm_shuffle_ps(m2.m[0], m2.m[0], _MM_SHUFFLE(2, 2, 2, 2));
    __m128 brod4 = _mm_shuffle_ps(m2.m[0], m2.m[0], _MM_SHUFFLE(3, 3, 3, 3));
    __m128 row   = _mm_add_ps(_mm_add_ps(
                                _mm_mul_ps(brod1, m1.m[0]),
                                _mm_mul_ps(brod2, m1.m[1])),
                              _mm_add_ps(
                                _mm_mul_ps(brod3, m1.m[2]),
                                _mm_mul_ps(brod4, m1.m[3])));
    res.m[0]     = row;

    brod1    = _mm_shuffle_ps(m2.m[1], m2.m[1], _MM_SHUFFLE(0, 0, 0, 0));
    brod2    = _mm_shuffle_ps(m2.m[1], m2.m[1], _MM_SHUFFLE(1, 1, 1, 1));
    brod3    = _mm_shuffle_ps(m2.m[1], m2.m[1], _MM_SHUFFLE(2, 2, 2, 2));
    brod4    = _mm_shuffle_ps(m2.m[1], m2.m[1], _MM_SHUFFLE(3, 3, 3, 3));
    row      = _mm_add_ps(_mm_add_ps(
                         _mm_mul_ps(brod1, m1.m[0]),
                         _mm_mul_ps(brod2, m1.m[1])),
                          _mm_add_ps(
                         _mm_mul_ps(brod3, m1.m[2]),
                         _mm_mul_ps(brod4, m1.m[3])));
    res.m[1] = row;

    brod1    = _mm_shuffle_ps(m2.m[2], m2.m[2], _MM_SHUFFLE(0, 0, 0, 0));
    brod2    = _mm_shuffle_ps(m2.m[2], m2.m[2], _MM_SHUFFLE(1, 1, 1, 1));
    brod3    = _mm_shuffle_ps(m2.m[2], m2.m[2], _MM_SHUFFLE(2, 2, 2, 2));
    brod4    = _mm_shuffle_ps(m2.m[2], m2.m[2], _MM_SHUFFLE(3, 3, 3, 3));
    row      = _mm_add_ps(_mm_add_ps(
                         _mm_mul_ps(brod1, m1.m[0]),
                         _mm_mul_ps(brod2, m1.m[1])),
                          _mm_add_ps(
                         _mm_mul_ps(brod3, m1.m[2]),
                         _mm_mul_ps(brod4, m1.m[3])));
    res.m[2] = row;

    brod1    = _mm_shuffle_ps(m2.m[3], m2.m[3], _MM_SHUFFLE(0, 0, 0, 0));
    brod2    = _mm_shuffle_ps(m2.m[3], m2.m[3], _MM_SHUFFLE(1, 1, 1, 1));
    brod3    = _mm_shuffle_ps(m2.m[3], m2.m[3], _MM_SHUFFLE(2, 2, 2, 2));
    brod4    = _mm_shuffle_ps(m2.m[3], m2.m[3], _MM_SHUFFLE(3, 3, 3, 3));
    row      = _mm_add_ps(_mm_add_ps(
                         _mm_mul_ps(brod1, m1.m[0]),
                         _mm_mul_ps(brod2, m1.m[1])),
                          _mm_add_ps(
                         _mm_mul_ps(brod3, m1.m[2]),
                         _mm_mul_ps(brod4, m1.m[3])));
    res.m[3] = row;
    return res;
#endif
}

MATEMATIKA_INLINE
mmat4 mate_mat_identity(void)
{
    mmat4 res;
    res.m[0] = _mm_setr_ps(1.0, 0.0, 0.0, 0.0);
    res.m[1] = _mm_setr_ps(0.0, 1.0, 0.0, 0.0);
    res.m[2] = _mm_setr_ps(0.0, 0.0, 1.0, 0.0);
    res.m[3] = _mm_setr_ps(0.0, 0.0, 0.0, 1.0);
    return res;
}

MATEMATIKA_INLINE
mmat4 mate_translation_make(const float x, const float y, const float z)
{
    mmat4 res = {0};
    res.m[0]  = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    res.m[1]  = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    res.m[2]  = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    res.m[3]  = _mm_setr_ps(x, y, z, 1.0f);
    return res;
}

MATEMATIKA_INLINE
mmat4 mate_scale_make(const float x, const float y, const float z)
{
    mmat4 res = {0};
    res.m[0]  = _mm_setr_ps(x, 0.0f, 0.0f, 0.0f);
    res.m[1]  = _mm_setr_ps(0.0f, y, 0.0f, 0.0f);
    res.m[2]  = _mm_setr_ps(0.0f, 0.0f, z, 0.0f);
    res.m[3]  = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    return res;
}

MATEMATIKA_INLINE
mmat4 dash_mat_copy(const mmat4 src)
{
    mmat4 res = {0};
    res.m[0]  = src.m[0];
    res.m[1]  = src.m[1];
    res.m[2]  = src.m[2];
    res.m[3]  = src.m[3];
    return res;
}

MATEMATIKA_INLINE
mmat4 mate_rotation_make(float x, float y, float z, const float angle)
{
    /* normalise x, y, z */
    float sq = sqrtf(x * x + y * y + z * z);
    x        = sq == 0.0f ? 0.0f : x / sq;
    y        = sq == 0.0f ? 0.0f : y / sq;
    z        = sq == 0.0f ? 0.0f : z / sq;

    mmat4       res = {0};
    const float c   = cosf(angle);
    const float s   = sinf(angle);
    const float t   = 1.0f - c;

    res.m[0] = _mm_setr_ps(c + x * x * t, /*******/ x * y * t + z * s, /**/ x * z * t - y * s, /**/ 0.0f);
    res.m[1] = _mm_setr_ps(x * y * t - z * s, /***/ c + y * y * t, /******/ y * z * t + x * s, /**/ 0.0f);
    res.m[2] = _mm_setr_ps(x * z * t + y * s, /***/ y * z * t - x * s, /**/ c + z * z * t, /******/ 0.0f);
    res.m[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    return res;
}

MATEMATIKA_INLINE
mmat4 mate_rotX_make(const float angle_rad)
{
    const float s = sinf(angle_rad);
    const float c = cosf(angle_rad);

    mmat4 res = {0};
    res.m[0]  = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    res.m[1]  = _mm_setr_ps(0.0f, c, s, 0.0f);
    res.m[2]  = _mm_setr_ps(0.0f, -s, c, 0.0f);
    res.m[3]  = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    return res;
}

MATEMATIKA_INLINE
mmat4 mate_rotY_make(const float angle_rad)
{
    const float s = sinf(angle_rad);
    const float c = cosf(angle_rad);

    mmat4 res = {0};
    res.m[0]  = _mm_setr_ps(c, 0.0f, s, 0.0f);
    res.m[1]  = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    res.m[2]  = _mm_setr_ps(-s, 0.0f, c, 0.0f);
    res.m[3]  = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    return res;
}

MATEMATIKA_INLINE
mmat4 mate_rotZ_make(const float angle_rad)
{
    const float s = sinf(angle_rad);
    const float c = cosf(angle_rad);

    mmat4 res = {0};
    res.m[0]  = _mm_setr_ps(c, -s, 0.0f, 0.0f);
    res.m[1]  = _mm_setr_ps(s, c, 0.0f, 0.0f);
    res.m[2]  = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    res.m[3]  = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    return res;
}

/**
 * @brief Creates a perspective projection matrix with a right-hand coordinate system and a
 *        clip-space of [-1, 1]. Based on the given parameters: field of view (`fov`), aspect ratio (`aspect`),
 *        near plane distance (`near`), and far plane distance (`far`).
 *
 * @param fov The hoprizontal field of view angle in degrees.
 * @param aspect The aspect ratio of the projection (width divided by height).
 * @param near The distance to the near clipping plane.
 * @param far The distance to the far clipping plane.
 * @return The resulting perspective projection matrix.
 */
MATEMATIKA_INLINE
mmat4 mate_perspective(const float fov, const float aspect, const float near, const float far)
{
    mmat4 res = {0};

    const float f = 1.0f / tanf(fov * 0.5f);

    res.m[0] = _mm_setr_ps(f / aspect, 0.0f, 0.0f, 0.0f);
    res.m[1] = _mm_setr_ps(0.0f, f, 0.0f, 0.0f);
    res.m[2] = _mm_setr_ps(0.0f, 0.0f, (far + near) / (near - far), -1.0f);
    res.m[3] = _mm_setr_ps(0.0f, 0.0f, (2.0f * far * near) / (near - far), 0.0f);

    return res;
}

/**
 * @brief Set up view matrix with RIGHT HANDED coordinate system
 *
 * @param eye eye vector
 * @param center center vector
 * @param up up vector
 * @return resulting lookat matrix
 */
MATEMATIKA_INLINE
mmat4 mate_look_at(const vec3 eye, const vec3 center, const vec3 up)
{
    const mvec4 eye4    = mate_cvt_vec3(eye, 0.0f);
    const mvec4 center4 = mate_cvt_vec3(center, 0.0f);
    const mvec4 up4     = mate_cvt_vec3(up, 0.0f);

    mvec4 sub     = {0};
    sub.m         = _mm_sub_ps(center4.m, eye4.m);
    const mvec4 f = mate_norm3(sub);
    const mvec4 s = mate_norm3(mate_cross(f, up4));
    const mvec4 u = mate_cross(s, f);

    mmat4 res   = {0};
    res.f[0][0] = s.f[0];
    res.f[0][1] = u.f[0];
    res.f[0][2] = -f.f[0];
    res.f[0][3] = 0.0f;

    res.f[1][0] = s.f[1];
    res.f[1][1] = u.f[1];
    res.f[1][2] = -f.f[1];
    res.f[1][3] = 0.0f;

    res.f[2][0] = s.f[2];
    res.f[2][1] = u.f[2];
    res.f[2][2] = -f.f[2];
    res.f[2][3] = 0.0f;

    res.f[3][0] = -mate_dot(s, eye4);
    res.f[3][1] = -mate_dot(u, eye4);
    res.f[3][2] = mate_dot(f, eye4);
    res.f[3][3] = 1.0f;

    return res;
}

MATEMATIKA_INLINE
void mate_mat_print(const mmat4 m)
{
    for (size_t i = 0; i < 4; i++)
    {
        // Convert the __m128 vector to an array of four floats
        float values[4]; // Align?
        _mm_store_ps(values, m.m[i]);

        printf("  | %f,  %f,  %f,  %f |\n", values[0], values[1], values[2], values[3]);
    }
}

#endif // __MAT4_H__