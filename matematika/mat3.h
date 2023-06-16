#ifndef __MAT3_H__
#define __MAT3_H__

#include "common.h"
#include "vec4.h"

typedef struct
{
    float f[3][3];
} mmat3;

MATEMATIKA_INLINE
mmat3 mate_mat3_scale(const mmat3 m, const float s)
{
    mmat3 res = {0};

    res.f[0][0] = m.f[0][0] * s;
    res.f[0][1] = m.f[0][1] * s;
    res.f[0][2] = m.f[0][2] * s;
    res.f[1][0] = m.f[1][0] * s;
    res.f[1][1] = m.f[1][1] * s;
    res.f[1][2] = m.f[1][2] * s;
    res.f[2][0] = m.f[2][0] * s;
    res.f[2][1] = m.f[2][1] * s;
    res.f[2][2] = m.f[2][2] * s;

    return res;
}

MATEMATIKA_INLINE
void mate_mat3_mulv(const mmat3 m, const vec3 v, vec3 dest)
{
    dest[0] = m.f[0][0] * v[0] + m.f[1][0] * v[1] + m.f[2][0] * v[2];
    dest[1] = m.f[0][1] * v[0] + m.f[1][1] * v[1] + m.f[2][1] * v[2];
    dest[2] = m.f[0][2] * v[0] + m.f[1][2] * v[1] + m.f[2][2] * v[2];
}

MATEMATIKA_INLINE
mmat3 mate_mat4_to_mat3(const mmat4 m)
{
    mmat3 res   = {0};
    res.f[0][0] = m.f[0][0];
    res.f[1][0] = m.f[1][0];
    res.f[2][0] = m.f[2][0];

    res.f[0][1] = m.f[0][1];
    res.f[1][1] = m.f[1][1];
    res.f[2][1] = m.f[2][1];

    res.f[0][2] = m.f[0][2];
    res.f[1][2] = m.f[1][2];
    res.f[2][2] = m.f[2][2];

    return res;
}

MATEMATIKA_INLINE
mmat3 mate_mat4_make_mat3_transpose(const mmat4 m)
{
    mmat3 res = {0};

    res.f[0][0] = m.f[0][0];
    res.f[0][1] = m.f[1][0];
    res.f[0][2] = m.f[2][0];
    res.f[1][0] = m.f[0][1];
    res.f[1][1] = m.f[1][1];
    res.f[1][2] = m.f[2][1];
    res.f[2][0] = m.f[0][2];
    res.f[2][1] = m.f[1][2];
    res.f[2][2] = m.f[2][2];

    return res;
}

MATEMATIKA_INLINE
mmat3 mate_mat3_transpose(const mmat3 m)
{
    mmat3 res = {0};

    res.f[0][0] = m.f[0][0];
    res.f[0][1] = m.f[1][0];
    res.f[0][2] = m.f[2][0];
    res.f[1][0] = m.f[0][1];
    res.f[1][1] = m.f[1][1];
    res.f[1][2] = m.f[2][1];
    res.f[2][0] = m.f[0][2];
    res.f[2][1] = m.f[1][2];
    res.f[2][2] = m.f[2][2];

    return res;
}

MATEMATIKA_INLINE
mmat3 mate_mat3_inv(mmat3 m)
{
    mmat3 res = {0};

    const float a = m.f[0][0], b = m.f[0][1], c = m.f[0][2],
                d = m.f[1][0], e = m.f[1][1], f = m.f[1][2],
                g = m.f[2][0], h = m.f[2][1], i = m.f[2][2];

    res.f[0][0] = e * i - f * h;
    res.f[0][1] = -(b * i - h * c);
    res.f[0][2] = b * f - e * c;
    res.f[1][0] = -(d * i - g * f);
    res.f[1][1] = a * i - c * g;
    res.f[1][2] = -(a * f - d * c);
    res.f[2][0] = d * h - g * e;
    res.f[2][1] = -(a * h - g * b);
    res.f[2][2] = a * e - b * d;

    const float det = 1.0f / (a * res.f[0][0] + b * res.f[1][0] + c * res.f[2][0]);

    return mate_mat3_scale(res, det);
}

// mmat3 mate_tbn_create(const mvec4 Tangent, const mvec4 Normal)
//{
//     mvec4 Bitangent = mate_cross(Normal, Tangent);

//    mmat3 TBN       = {0};
//    TBN.elements[0] = Tangent.m128_f32[0];
//    TBN.elements[1] = Bitangent.m128_f32[0];
//    TBN.elements[2] = Normal.m128_f32[0];
//    TBN.elements[3] = Tangent.m128_f32[1];
//    TBN.elements[4] = Bitangent.m128_f32[1];
//    TBN.elements[5] = Normal.m128_f32[1];
//    TBN.elements[6] = Tangent.m128_f32[2];
//    TBN.elements[7] = Bitangent.m128_f32[2];
//    TBN.elements[8] = Normal.m128_f32[2];

//    return TBN;
//}

#endif // __MAT3_H__