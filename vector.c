#include "vector.h"

Mat4x4 Get_Rotation_X_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0.0f};

    matrix.matrix[0][0] = 1.0f;
    matrix.matrix[1][1] = cosf(angle_rad);
    matrix.matrix[1][2] = sinf(angle_rad);
    matrix.matrix[2][1] = -sinf(angle_rad);
    matrix.matrix[2][2] = cosf(angle_rad);
    matrix.matrix[3][3] = 1.0f;

    return matrix;
}

Mat4x4 Get_Rotation_Y_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0.0f};

    matrix.matrix[0][0] = cosf(angle_rad);
    matrix.matrix[0][2] = sinf(angle_rad);
    matrix.matrix[1][1] = 1.0f;
    matrix.matrix[2][0] = -sinf(angle_rad);
    matrix.matrix[2][2] = cosf(angle_rad);
    matrix.matrix[3][3] = 1.0f;

    return matrix;
}

Mat4x4 Get_Rotation_Z_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0.0f};

    matrix.matrix[0][0] = cosf(angle_rad);
    matrix.matrix[0][1] = -sinf(angle_rad);
    matrix.matrix[1][0] = sinf(angle_rad);
    matrix.matrix[1][1] = cosf(angle_rad);
    matrix.matrix[2][2] = 1.0f;
    matrix.matrix[3][3] = 1.0f;

    return matrix;
}

Mat4x4 Get_Translation_Matrix(float x, float y, float z)
{
    Mat4x4 matrix = {0.0f};

    matrix.matrix[0][0] = 1.0f;
    matrix.matrix[1][1] = 1.0f;
    matrix.matrix[2][2] = 1.0f;
    matrix.matrix[3][3] = 1.0f;
    matrix.matrix[3][0] = x;
    matrix.matrix[3][1] = y;
    matrix.matrix[3][2] = z;

    return matrix;
}

Mat4x4 Get_Projection_Matrix(float FOV_Degrees, float aspect_ratio, float near, float far)
{
    const float fFovRad = 1.0f / tanf(FOV_Degrees * 0.5f / 180.0f * 3.14159f);

    Mat4x4 matrix = {0.0f};

    matrix.matrix[0][0] = aspect_ratio * fFovRad;
    matrix.matrix[1][1] = fFovRad;
    matrix.matrix[2][2] = far / (far - near);
    matrix.matrix[3][2] = (-far * near) / (far - near);
    matrix.matrix[2][3] = 1.0f;
    matrix.matrix[3][3] = 0.0f;

    return matrix;
}

void Matrix_Multiply_Matrix(const float *A, const float *B, float *C)
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
        _mm_store_ps(&C[4 * i], row);
    }
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

vec4 Vector_Add(const vec4 *v1, const vec4 *v2)
{
    return (vec4){v1->coords.x + v2->coords.x, v1->coords.y + v2->coords.y, v1->coords.z + v2->coords.z, v1->coords.w + v2->coords.w};
}

vec4 Vector_Sub(const vec4 *v1, const vec4 *v2)
{
    return (vec4){v1->coords.x - v2->coords.x, v1->coords.y - v2->coords.y, v1->coords.z - v2->coords.z, v1->coords.w - v2->coords.w};
}

vec4 Vector_Mul(const vec4 *v1, float k)
{
    return (vec4){v1->coords.x * k, v1->coords.y * k, v1->coords.z * k, v1->coords.w * k};
}

vec4 Vector_Div(const vec4 *v1, float k)
{
    return (vec4){v1->coords.x / k, v1->coords.y / k, v1->coords.z / k, v1->coords.w / k};
}
