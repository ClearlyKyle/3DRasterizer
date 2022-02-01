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

Mat4x4 Get_Rotation_X_Matrix(float angle_rad);
Mat4x4 Get_Rotation_Y_Matrix(float angle_rad);
Mat4x4 Get_Rotation_Z_Matrix(float angle_rad);
Mat4x4 Get_Translation_Matrix(float x, float y, float z);
Mat4x4 Get_Projection_Matrix(float FOV_Degrees, float aspect_ratio, float near, float far);

void Matrix_Multiply_Matrix(const float *A, const float *B, float *C);
void Vector_Cross_Product(const float *v0, const float *v1, float *output);

vec4 Vector_Add(const vec4 *v1, const vec4 *v2);
vec4 Vector_Sub(const vec4 *v1, const vec4 *v2);
vec4 Vector_Mul(const vec4 *v1, float k);
vec4 Vector_Div(const vec4 *v1, float k);

#endif // __VECTOR_H__