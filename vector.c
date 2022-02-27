#include "vector.h"

Mat4x4 Get_Translation_Matrix(float x, float y, float z)
{
    Mat4x4 matrix = {0.0f};

    matrix.elements[0] = 1.0f;
    matrix.elements[5] = 1.0f;
    matrix.elements[10] = 1.0f;
    matrix.elements[15] = 1.0f;
    matrix.elements[12] = x;
    matrix.elements[13] = y;
    matrix.elements[14] = z;

    return matrix;
}

Mat4x4 Get_Rotation_X_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0.0f};

    matrix.elements[0] = 1.0f;
    matrix.elements[5] = cosf(angle_rad);
    matrix.elements[6] = sinf(angle_rad);
    matrix.elements[9] = -sinf(angle_rad);
    matrix.elements[10] = cosf(angle_rad);
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4 Get_Rotation_Y_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0.0f};

    matrix.elements[0] = cosf(angle_rad);
    matrix.elements[2] = sinf(angle_rad);
    matrix.elements[5] = 1.0f;
    matrix.elements[8] = -sinf(angle_rad);
    matrix.elements[10] = cosf(angle_rad);
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4 Get_Rotation_Z_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0.0f};

    matrix.elements[0] = cosf(angle_rad);
    matrix.elements[1] = -sinf(angle_rad);
    matrix.elements[4] = sinf(angle_rad);
    matrix.elements[5] = cosf(angle_rad);
    matrix.elements[10] = 1.0f;
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4_m128 Get_Translation_Matrix_m128(float x, float y, float z)
{
    // Mat4x4 matrix = {0.0f};

    // matrix.elements[0] = 1.0f;
    // matrix.elements[5] = 1.0f;
    // matrix.elements[10] = 1.0f;
    // matrix.elements[15] = 1.0f;
    // matrix.elements[12] = x;
    // matrix.elements[13] = y;
    // matrix.elements[14] = z;

    // 3  2  1  0
    // 7  6  5  4
    // 11 10 9  8
    // 15 14 13 12

    //        [0]         [1]         [2]         [3]
    // test = {1.00000000, 2.00000000, 3.00000000, 4.00000000}
    // const __m128 test = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);

    Mat4x4_m128 m;
    m.rows[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    m.rows[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
    m.rows[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    m.rows[3] = _mm_set_ps(1.0f, z, y, x);
    return m;
}

Mat4x4_m128 Get_Rotation_X_Matrix_m128(float angle_rad)
{
    Mat4x4_m128 m;
    m.rows[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    m.rows[1] = _mm_set_ps(0.0f, sinf(angle_rad), cosf(angle_rad), 0.0f);
    m.rows[2] = _mm_set_ps(0.0f, cosf(angle_rad), -sinf(angle_rad), 0.0f);
    m.rows[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return m;
}

Mat4x4_m128 Get_Rotation_Y_Matrix_m128(float angle_rad)
{
    Mat4x4_m128 m;
    m.rows[0] = _mm_set_ps(0.0f, sinf(angle_rad), 0.0f, cosf(angle_rad));
    m.rows[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
    m.rows[2] = _mm_set_ps(0.0f, cosf(angle_rad), 0.0f, -sinf(angle_rad));
    m.rows[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return m;
}

Mat4x4_m128 Get_Rotation_Z_Matrix_m128(float angle_rad)
{
    Mat4x4_m128 m;
    m.rows[0] = _mm_set_ps(0.0f, 0.0f, -sinf(angle_rad), cosf(angle_rad));
    m.rows[1] = _mm_set_ps(0.0f, 0.0f, cosf(angle_rad), sinf(angle_rad));
    m.rows[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    m.rows[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return m;
}

Mat4x4 Get_Projection_Matrix(float FOV_Degrees, float aspect_ratio, float near, float far)
{
    const float fFovRad = 1.0f / tanf(FOV_Degrees * 0.5f / 180.0f * 3.14159f);

    Mat4x4 matrix = {0.0f};

    matrix.elements[0] = aspect_ratio * fFovRad;
    matrix.elements[5] = fFovRad;
    matrix.elements[10] = far / (far - near);
    matrix.elements[14] = (-far * near) / (far - near);
    matrix.elements[11] = 1.0f;
    matrix.elements[15] = 0.0f;

    return matrix;
}

Mat4x4_m128 Get_Projection_Matrix_m128(float FOV_Degrees, float aspect_ratio, float near, float far)
{
    const float fFovRad = 1.0f / tanf(FOV_Degrees * 0.5f / 180.0f * 3.14159f);

    Mat4x4_m128 m;
    m.rows[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, aspect_ratio * fFovRad);
    m.rows[1] = _mm_set_ps(0.0f, 0.0f, fFovRad, 0.0f);
    m.rows[2] = _mm_set_ps(1.0f, far / (far - near), 0.0f, 0.0f);
    m.rows[3] = _mm_set_ps(0.0f, (-far * near) / (far - near), 0.0f, 0.0f);
    return m;
}

void Matrix_Multiply_Vector(const float *M, const float *vec, float *output)
{
    __m128 brod1 = _mm_load_ps(vec);
    __m128 brod2 = _mm_load_ps(vec);
    __m128 brod3 = _mm_load_ps(vec);
    __m128 brod4 = _mm_load_ps(vec);
    __m128 row = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(brod1, _mm_load_ps(&M[0])),
            _mm_mul_ps(brod2, _mm_load_ps(&M[4]))),
        _mm_add_ps(
            _mm_mul_ps(brod3, _mm_load_ps(&M[8])),
            _mm_mul_ps(brod4, _mm_load_ps(&M[12]))));

    _mm_store_ps(output, row);
}

__m128 Matrix_Multiply_Vector_m128(const Mat4x4_m128 *M, const __m128 vec)
{
    const __m128 brod1 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 brod2 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 brod3 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 brod4 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));

    return _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(brod1, M->rows[0]),
            _mm_mul_ps(brod2, M->rows[1])),
        _mm_add_ps(
            _mm_mul_ps(brod3, M->rows[2]),
            _mm_mul_ps(brod4, M->rows[3])));
}

__m128 Matrix_Multiply_Vector_SIMD(const float *M, const __m128 vec)
{
    // Do we need to store these?
    const __m128 brod1 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 brod2 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 brod3 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 brod4 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));

    return _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(brod1, _mm_load_ps(&M[0])),
            _mm_mul_ps(brod2, _mm_load_ps(&M[4]))),
        _mm_add_ps(
            _mm_mul_ps(brod3, _mm_load_ps(&M[8])),
            _mm_mul_ps(brod4, _mm_load_ps(&M[12]))));
}

void Matrix_Multiply_Matrix(const float *A, const float *B, float *Output_Matrix)
{
    __m128 row1 = _mm_load_ps(&B[0]);
    __m128 row2 = _mm_load_ps(&B[4]);
    __m128 row3 = _mm_load_ps(&B[8]);
    __m128 row4 = _mm_load_ps(&B[12]);

    for (int i = 0; i < 4; i++)
    {
        __m128 brod1 = _mm_set1_ps(A[4 * i + 0]);
        __m128 brod2 = _mm_set1_ps(A[4 * i + 1]);
        __m128 brod3 = _mm_set1_ps(A[4 * i + 2]);
        __m128 brod4 = _mm_set1_ps(A[4 * i + 3]);
        __m128 row = _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(brod1, row1),
                _mm_mul_ps(brod2, row2)),
            _mm_add_ps(
                _mm_mul_ps(brod3, row3),
                _mm_mul_ps(brod4, row4)));
        _mm_store_ps(&Output_Matrix[4 * i], row);
    }
}

Mat4x4_m128 Matrix_Multiply_Matrix_m128(const Mat4x4_m128 *A, const Mat4x4_m128 *B)
{
    Mat4x4_m128 res;
    Mat4x4_m128 tmp = *A; // Is this needed?

    _MM_TRANSPOSE4_PS(tmp.rows[0], tmp.rows[1], tmp.rows[2], tmp.rows[3]);

    for (int i = 0; i < 4; i++)
    {
        res.rows[i] = _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(_mm_shuffle_ps(tmp.rows[i], tmp.rows[i], _MM_SHUFFLE(0, 0, 0, 0)), B->rows[0]),
                _mm_mul_ps(_mm_shuffle_ps(tmp.rows[i], tmp.rows[i], _MM_SHUFFLE(1, 1, 1, 1)), B->rows[1])),
            _mm_add_ps(
                _mm_mul_ps(_mm_shuffle_ps(tmp.rows[i], tmp.rows[i], _MM_SHUFFLE(2, 2, 2, 2)), B->rows[2]),
                _mm_mul_ps(_mm_shuffle_ps(tmp.rows[i], tmp.rows[i], _MM_SHUFFLE(3, 3, 3, 3)), B->rows[3])));
    }
    return res;
}

void Vector_Cross_Product(const float *v0, const float *v1, float *output)
{
    __m128 vec0 = _mm_load_ps(v0);
    __m128 vec1 = _mm_load_ps(v1);

    __m128 tmp0 = _mm_shuffle_ps(vec0, vec0, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 tmp1 = _mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 tmp2 = _mm_mul_ps(tmp0, vec1);
    __m128 tmp3 = _mm_mul_ps(tmp0, tmp1);
    __m128 tmp4 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 result = _mm_sub_ps(tmp3, tmp4);

    _mm_store_ps(output, result);
}
__m128 Vector_Cross_Product_SIMD(const __m128 vec0, const __m128 vec1)
{
    const __m128 tmp0 = _mm_shuffle_ps(vec0, vec0, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 tmp1 = _mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(3, 1, 0, 2));
    const __m128 tmp2 = _mm_mul_ps(tmp0, vec1);
    const __m128 tmp3 = _mm_mul_ps(tmp0, tmp1);
    const __m128 tmp4 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3, 0, 2, 1));
    return _mm_sub_ps(tmp3, tmp4);
}

void Calculate_Surface_Normal(const float *A, const float *B, const float *C, float *output)
{
    __m128 vertex1 = _mm_load_ps(A);
    __m128 vertex2 = _mm_load_ps(B);
    __m128 vertex3 = _mm_load_ps(C);

    __m128 normal = _mm_mul_ps(
        _mm_sub_ps(vertex2, vertex1),
        _mm_sub_ps(vertex3, vertex1));

    normal = _mm_mul_ps(normal, normal);

    // 2 hadd's are slow...
    // https://stackoverflow.com/questions/4120681/how-to-calculate-single-vector-dot-product-using-sse-intrinsic-functions-in-c
    //__m128 sum = _mm_hadd_ps(_mm_hadd_ps(normal, _mm_setzero_ps()), _mm_setzero_ps());

    // normal = _mm_div_ps(normal, _mm_sqrt_ps(sum));

    _mm_store_ps(output, normal);
}

__m128 Calculate_Surface_Normal_m128(const __m128 A, const __m128 B, const __m128 C)
{
    const __m128 normal = _mm_mul_ps(
        _mm_sub_ps(B, A),
        _mm_sub_ps(C, A));

    return _mm_mul_ps(normal, normal);
}

float hsum_ps_sse3(const __m128 v)
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

__m128 Calculate_Surface_Normal_SIMD(const __m128 v1, const __m128 v2, const __m128 v3)
{
    const __m128 cross_product_result = Vector_Cross_Product_SIMD(_mm_sub_ps(v1, v2), _mm_sub_ps(v1, v3));

    const __m128 sqrt_result = _mm_sqrt_ps(_mm_set1_ps(hsum_ps_sse3(_mm_mul_ps(cross_product_result, cross_product_result))));

    return _mm_mul_ps(sqrt_result, cross_product_result);
}

float Calculate_Dot_Product_SIMD(const __m128 v1, const __m128 v2)
{
    return _mm_cvtss_f32(_mm_dp_ps(v1, v2, 0xf1));
}

__m128 Normalize_m128(__m128 input)
{
    input.m128_f32[3] = 0.0f;
    const __m128 squared = _mm_mul_ps(input, input); // square the input values

    const float sqr_sum = hsum_ps_sse3(squared);
    const __m128 tmp_inv_sqrt = _mm_invsqrt_ps(_mm_set1_ps(sqr_sum));

    return _mm_mul_ps(input, tmp_inv_sqrt);
}

__m128 Clamp_m128(const __m128 vec, float minval, float maxval)
{
    // Branchless SSE clamp.
    return _mm_min_ps(_mm_max_ps(vec, _mm_set1_ps(minval)), _mm_set1_ps(maxval));
}

// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
// I - 2.0 * dot(N, I) * N
__m128 Reflect_m128(const __m128 I, const __m128 N)
{
    __m128 righ = _mm_mul_ps(_mm_set1_ps(Calculate_Dot_Product_SIMD(N, I)), N);
    righ = _mm_mul_ps(_mm_set1_ps(2.0f), righ);
    return _mm_mul_ps(I, righ);
}