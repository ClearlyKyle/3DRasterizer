#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <math.h>
#include <xmmintrin.h>

typedef union vec4_u
{
    float points[4];
    struct
    {
        float x, y, z, w;
    } coords;
} vec4;

typedef union vec3_u
{
    float points[3];
    struct
    {
        float x, y, z;
    } coords;
} vec3;

typedef union Mat4x4_u
{
    // 1  2  3  4
    // 5  6  7  8
    // 9  10 11 12
    // 13 14 15 16
    float elements[16];
    float matrix[4][4];
} Mat4x4;

struct Mesh
{
    unsigned int count;

    float *x;
    float *y;
    float *z;
    float *w;
};

#endif // __VECTOR_H__