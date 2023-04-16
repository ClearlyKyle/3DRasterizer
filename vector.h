#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <math.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

typedef struct
{
    // 0  1  2  3
    // 4  5  6  7
    // 8  9  10 11
    // 12 13 14 15
    float elements[16];
} Mat4x4;

typedef struct
{
    // 0  1  2
    // 3  4  5
    // 6  7  8
    float elements[9];
} Mat3x3;

typedef struct
{
    // 4 3 2 1
    __m128 rows[4];
} Mat4x4_m128;

typedef struct Mesh_s
{
    unsigned int count;

    float *x;
    float *y;
    float *z;
    float *w;
    float *u;
    float *v;
} Mesh;

// MACROS
#define M_PI                       3.14159265358979323846264338327950288
#define DEG_to_RAD(angleInDegrees) ((angleInDegrees)*M_PI / 180.0)
#define RAD_to_DEG(angleInRadians) ((angleInRadians)*180.0 / M_PI)

// FUNCTION HEADERS
Mat4x4 Get_Rotation_X_Matrix(float angle_rad);
Mat4x4 Get_Rotation_Y_Matrix(float angle_rad);
Mat4x4 Get_Rotation_Z_Matrix(float angle_rad);
Mat4x4 Get_Translation_Matrix(float x, float y, float z);
Mat4x4 Get_Projection_Matrix(float FOV_Degrees, float aspect_ratio, float near, float far);

Mat4x4_m128 Get_Translation_Matrix_m128(float x, float y, float z);
Mat4x4_m128 Get_Rotation_X_Matrix_m128(float angle_rad);
Mat4x4_m128 Get_Rotation_Y_Matrix_m128(float angle_rad);
Mat4x4_m128 Get_Rotation_Z_Matrix_m128(float angle_rad);
Mat4x4_m128 Get_Projection_Matrix_m128(float FOV_Degrees, float aspect_ratio, float near, float far);

Mat4x4 CreateViewMatrix(const float cameraPosition[3]);

void   Matrix_Multiply_Vector(const float *M, const float *vec, float *output);
__m128 Matrix_Multiply_Vector_m128(const Mat4x4_m128 *M, const __m128 vec);
__m128 Matrix_Multiply_Vector_SIMD(const float *M, const __m128 vec);

Mat4x4_m128 Matrix_Multiply_Matrix_m128(const Mat4x4_m128 *A, const Mat4x4_m128 *B);
__m128      Vector_Cross_Product_m128(const __m128 vec0, const __m128 vec1);
__m128      Calculate_Surface_Normal_m128(const __m128 A, const __m128 B, const __m128 C);

void   Matrix_Multiply_Matrix(const float *A, const float *B, float *Output_Matrix);
void   Vector_Cross_Product(const float *v0, const float *v1, float *output);
void   Calculate_Surface_Normal(const float *A, const float *B, const float *C, float *output);
__m128 Calculate_Surface_Normal_SIMD(const __m128 v1, const __m128 v2, const __m128 v3);
float  Calculate_Dot_Product_SIMD(const __m128 v1, const __m128 v2);

__m128 Normalize_m128(const __m128 intput);
__m128 Clamp_m128(const __m128 vec, const float minval, const float maxval);
float  hsum_ps_sse3(const __m128 v);
Mat4x4 Get_TBN_Matrix(__m128 Tangent, const __m128 Normal, const Mat4x4 ViewModelMatrix);

Mat3x3 Mat4x4_to_Mat3x3(const Mat4x4 m);

Mat3x3 Inverse_Transpose_Mat4x4_to_Mat3x3(const Mat4x4 worldMatrix);
Mat3x3 Create_TBN(const __m128 Tangent, const __m128 Normal);
Mat3x3 TransposeMat3x3(const Mat3x3 matrix);

Mat4x4 TransposeMat4x4(Mat4x4 *mat);
Mat4x4 InverseMat4x4(Mat4x4 *mat);

static __m128 Mat3x3_mul_m128(const Mat3x3 m, const __m128 v)
{
    float arr_v[4] = {0};
    _mm_storeu_ps(arr_v, v);

    __m128 result = _mm_setzero_ps();
    for (int i = 0; i < 3; i++)
    {
        __m128 row = _mm_loadu_ps(m.elements + 3 * i);
        __m128 val = _mm_set1_ps(arr_v[i]);
        result     = _mm_add_ps(result, _mm_mul_ps(row, val));
    }

    return result;
}

#endif // __VECTOR_H__