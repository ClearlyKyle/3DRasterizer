#ifndef __VEC3_H__
#define __VEC3_H__

#include "common.h"

typedef float vec3[3];

MATEMATIKA_INLINE
void mate_vec3_sub(const vec3 a, const vec3 b, vec3 dest)
{
    dest[0] = a[0] - b[0];
    dest[1] = a[1] - b[1];
    dest[2] = a[2] - b[2];
}

MATEMATIKA_INLINE
void mate_vec3_negate(vec3 v)
{
    v[0] = -v[0];
    v[1] = -v[1];
    v[2] = -v[2];
}

MATEMATIKA_INLINE
float mate_vec3_dot(const vec3 a, const vec3 b)
{
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

MATEMATIKA_INLINE
void mate_vec3_scale(const vec3 v, const float s, vec3 dest)
{
    dest[0] = v[0] * s;
    dest[1] = v[1] * s;
    dest[2] = v[2] * s;
}

MATEMATIKA_INLINE
void mate_vec3_normalise(vec3 v)
{
    const float norm = sqrtf(mate_vec3_dot(v, v));

    if (norm == 0.0f)
    {
        v[0] = v[1] = v[2] = 0.0f;
        return;
    }

    mate_vec3_scale(v, 1.0f / norm, v);
}

MATEMATIKA_INLINE
void mate_vec3_cross(const vec3 a, const vec3 b, vec3 dest)
{
    dest[0] = a[1] * b[2] - a[2] * b[1];
    dest[1] = a[2] * b[0] - a[0] * b[2];
    dest[2] = a[0] * b[1] - a[1] * b[0];
}

#endif // __VEC3_H__