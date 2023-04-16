#include "vector.h"

// Column Major
Mat4x4 Get_Translation_Matrix(float x, float y, float z)
{
    Mat4x4 matrix = {0};

    matrix.elements[0]  = 1.0f;
    matrix.elements[5]  = 1.0f;
    matrix.elements[10] = 1.0f;
    matrix.elements[15] = 1.0f;
    matrix.elements[12] = x;
    matrix.elements[13] = y;
    matrix.elements[14] = z;

    return matrix;
}

Mat4x4 Get_Rotation_X_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0};

    matrix.elements[0]  = 1.0f;
    matrix.elements[5]  = cosf(angle_rad);
    matrix.elements[6]  = -sinf(angle_rad);
    matrix.elements[9]  = -sinf(angle_rad);
    matrix.elements[10] = cosf(angle_rad);
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4 Get_Rotation_Y_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0};

    matrix.elements[0]  = cosf(angle_rad);
    matrix.elements[2]  = -sinf(angle_rad);
    matrix.elements[5]  = 1.0f;
    matrix.elements[8]  = sinf(angle_rad);
    matrix.elements[10] = cosf(angle_rad);
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4 Get_Rotation_Z_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0};

    matrix.elements[0]  = cosf(angle_rad);
    matrix.elements[1]  = sinf(angle_rad);
    matrix.elements[4]  = -sinf(angle_rad);
    matrix.elements[5]  = cosf(angle_rad);
    matrix.elements[10] = 1.0f;
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4 Get_View_Matrix(float angle_rad)
{
    Mat4x4 matrix = {0};
    // 0  1  2  3
    // 4  5  6  7
    // 8  9  10 11
    // 12 13 14 15
    // RIGHT
    matrix.elements[0]  = 1.0f;
    matrix.elements[4]  = 0.0f;
    matrix.elements[8]  = 0.0f;
    matrix.elements[12] = 0.0f;

    // UP
    matrix.elements[1]  = 0.0f;
    matrix.elements[5]  = 1.0f;
    matrix.elements[9]  = 0.0f;
    matrix.elements[13] = 0.0f;
    // FORWARD
    matrix.elements[2]  = 0.0f;
    matrix.elements[6]  = 0.0f;
    matrix.elements[10] = 1.0f;
    matrix.elements[14] = 0.0f;
    // POSITION
    matrix.elements[3]  = 0.0f;
    matrix.elements[7]  = 0.0f;
    matrix.elements[11] = 0.0f;
    matrix.elements[15] = 0.0f;

    matrix.elements[10] = 1.0f;
    matrix.elements[15] = 1.0f;

    return matrix;
}

Mat4x4 Get_Projection_Matrix(float FOV_Degrees, float aspect_ratio, float near, float far)
{
    const float fFovRad = 1.0f / tanf(FOV_Degrees * 0.5f / 180.0f * 3.14159f);

    Mat4x4 matrix = {0};

    matrix.elements[0]  = aspect_ratio * fFovRad;
    matrix.elements[5]  = fFovRad;
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
    __m128 row   = _mm_add_ps(
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

// Column Major Matrix Multiplication
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
        __m128 row   = _mm_add_ps(
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

    __m128 tmp0   = _mm_shuffle_ps(vec0, vec0, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 tmp1   = _mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 tmp2   = _mm_mul_ps(tmp0, vec1);
    __m128 tmp3   = _mm_mul_ps(tmp0, tmp1);
    __m128 tmp4   = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3, 0, 2, 1));
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
    // Compute the vectors AB and AC
    __m128 AB = _mm_sub_ps(B, A);
    __m128 AC = _mm_sub_ps(C, A);

    // Compute the cross product of AB and AC to get the surface normal
    __m128 cross = _mm_sub_ps(
        _mm_mul_ps(_mm_shuffle_ps(AB, AB, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(AC, AC, _MM_SHUFFLE(3, 1, 0, 2))),
        _mm_mul_ps(_mm_shuffle_ps(AB, AB, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(AC, AC, _MM_SHUFFLE(3, 0, 2, 1))));

    // Normalize the surface normal
    __m128 len_squared = _mm_dp_ps(cross, cross, 0x7f);
    __m128 inv_len     = _mm_rsqrt_ps(len_squared);
    __m128 normal      = _mm_mul_ps(cross, inv_len);

    return normal;
}

float hsum_ps_sse3(const __m128 v)
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// float hsum_ps_sse3(const __m128 v)
//{
//     __m128 sum = _mm_add_ps(v, _mm_movehl_ps(v, v)); // add high and low halves
//     sum        = _mm_hadd_ps(sum, sum);              // horizontally add the results
//     sum        = _mm_hadd_ps(sum, sum);
//     return _mm_cvtss_f32(sum);
// }

//__m128 Calculate_Surface_Normal_SIMD(const __m128 v1, const __m128 v2, const __m128 v3)
//{
//    //const __m128 cross_product_result = Vector_Cross_Product_SIMD(_mm_sub_ps(v1, v2), _mm_sub_ps(v1, v3));

//    //const __m128 sqrt_result = _mm_sqrt_ps(_mm_set1_ps(hsum_ps_sse3(_mm_mul_ps(cross_product_result, cross_product_result))));

//    //return _mm_mul_ps(sqrt_result, cross_product_result);

//}

//__m128 Calculate_Surface_Normal_m128(const __m128 A, const __m128 B, const __m128 C)
//{
//    const __m128 normal = _mm_mul_ps(
//        _mm_sub_ps(B, A),
//        _mm_sub_ps(C, A));

//    return _mm_mul_ps(normal, normal);
//}

__m128 Calculate_Surface_Normal_SIMD(const __m128 v1, const __m128 v2, const __m128 v3)
{
    // Calculate the edge vectors
    __m128 e1 = _mm_sub_ps(v2, v1);
    __m128 e2 = _mm_sub_ps(v3, v1);

    // Calculate the cross product of the edge vectors
    __m128 cross = _mm_sub_ps(
        _mm_mul_ps(_mm_shuffle_ps(e1, e1, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(e2, e2, _MM_SHUFFLE(3, 1, 0, 2))),
        _mm_mul_ps(_mm_shuffle_ps(e1, e1, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(e2, e2, _MM_SHUFFLE(3, 0, 2, 1))));

    // Normalize the cross product
    __m128 length = _mm_sqrt_ps(_mm_dp_ps(cross, cross, 0x7F));
    __m128 normal = _mm_div_ps(cross, length);

    return normal;
}

float Calculate_Dot_Product_SIMD(const __m128 v1, const __m128 v2)
{
    return _mm_cvtss_f32(_mm_dp_ps(v1, v2, 0x7f));
}

__m128 Normalize_m128(const __m128 input)
{
    __m128 squared      = _mm_mul_ps(input, input); // square the input values
    squared.m128_f32[3] = 0.0f;

    const float sqr_sum = squared.m128_f32[0] + squared.m128_f32[1] + squared.m128_f32[2];
    const float sqrt    = sqrtf(sqr_sum);
    
    return _mm_div_ps(input, _mm_set1_ps(sqrt));
}

__m128 Clamp_m128(const __m128 vec, const float minval, const float maxval)
{
    // Branchless SSE clamp.
    return _mm_min_ps(_mm_max_ps(vec, _mm_set1_ps(minval)), _mm_set1_ps(maxval));
}

Mat4x4 Get_TBN_Matrix(__m128 Tangent, __m128 Normal, const Mat4x4 ViewModelMatrix)
{
    Mat4x4 TBN = {0.0f};

    // Equation bellow : T = normalize(T - dot(T, N) * N);
    // const __m128 magic_tangent = _mm_mul_ps(_mm_set1_ps(Calculate_Dot_Product_SIMD(Tangent, Normal)), Normal);
    // Tangent = Normalize_m128(_mm_sub_ps(Tangent, magic_tangent));

    Tangent = Matrix_Multiply_Vector_SIMD(ViewModelMatrix.elements, Tangent);
    Normal  = Matrix_Multiply_Vector_SIMD(ViewModelMatrix.elements, Normal);

    //__m128 Bitangent = Vector_Cross_Product_SIMD(Tangent, Normal);
    __m128 Bitangent = Vector_Cross_Product_SIMD(Normal, Tangent);

    __m128 Zeros = _mm_setzero_ps();

    _MM_TRANSPOSE4_PS(Tangent, Bitangent, Normal, Zeros);

    _mm_store_ps(&TBN.elements[0], Tangent);
    _mm_store_ps(&TBN.elements[4], Bitangent);
    _mm_store_ps(&TBN.elements[8], Normal);
    _mm_store_ps(&TBN.elements[12], Zeros);

    TBN.elements[3]  = 0.0f;
    TBN.elements[7]  = 0.0f;
    TBN.elements[11] = 0.0f;
    TBN.elements[15] = 1.0f;

    return TBN;
}

Mat3x3 InverseMat3x3FromMat4x4(const Mat4x4 mat4x4)
{
    Mat3x3       mat3x3;
    const float *m = mat4x4.elements;

    float det = m[0] * (m[5] * m[10] - m[9] * m[6]) -
                m[1] * (m[4] * m[10] - m[6] * m[8]) +
                m[2] * (m[4] * m[9] - m[5] * m[8]);

    float inv_det = 1.0f / det;

    mat3x3.elements[0] = inv_det * (m[5] * m[10] - m[9] * m[6]);
    mat3x3.elements[1] = inv_det * (m[2] * m[9] - m[1] * m[10]);
    mat3x3.elements[2] = inv_det * (m[1] * m[6] - m[2] * m[5]);
    mat3x3.elements[3] = inv_det * (m[6] * m[8] - m[4] * m[10]);
    mat3x3.elements[4] = inv_det * (m[0] * m[10] - m[2] * m[8]);
    mat3x3.elements[5] = inv_det * (m[2] * m[4] - m[0] * m[6]);
    mat3x3.elements[6] = inv_det * (m[4] * m[9] - m[5] * m[8]);
    mat3x3.elements[7] = inv_det * (m[1] * m[8] - m[0] * m[9]);
    mat3x3.elements[8] = inv_det * (m[0] * m[5] - m[1] * m[4]);

    return mat3x3;
}

Mat4x4 CreateViewMatrix(const float cameraPosition[3])
{
    Mat4x4 viewMatrix;

    // Default up vector
    const float up[3] = {0.0f, 1.0f, 0.0f};

    // Default target
    const float target[3] = {0.0f, 0.0f, 0.0f};

    // Calculate the view direction
    const float viewDirection[3] = {target[0] - cameraPosition[0], target[1] - cameraPosition[1], target[2] - cameraPosition[2]};

    // Normalize the view direction
    const float viewDirectionLength        = sqrtf(viewDirection[0] * viewDirection[0] + viewDirection[1] * viewDirection[1] + viewDirection[2] * viewDirection[2]);
    const float viewDirectionNormalized[3] = {viewDirection[0] / viewDirectionLength, viewDirection[1] / viewDirectionLength, viewDirection[2] / viewDirectionLength};

    // Calculate the right vector
    const float right[3] = {up[1] * viewDirectionNormalized[2] - up[2] * viewDirectionNormalized[1], up[2] * viewDirectionNormalized[0] - up[0] * viewDirectionNormalized[2], up[0] * viewDirectionNormalized[1] - up[1] * viewDirectionNormalized[0]};

    // Calculate the new up vector
    const float newUp[3] = {viewDirectionNormalized[1] * right[2] - viewDirectionNormalized[2] * right[1], viewDirectionNormalized[2] * right[0] - viewDirectionNormalized[0] * right[2], viewDirectionNormalized[0] * right[1] - viewDirectionNormalized[1] * right[0]};

    // Set the view matrix elements
    viewMatrix.elements[0]  = right[0];
    viewMatrix.elements[1]  = newUp[0];
    viewMatrix.elements[2]  = -viewDirectionNormalized[0];
    viewMatrix.elements[3]  = 0.0f;
    viewMatrix.elements[4]  = right[1];
    viewMatrix.elements[5]  = newUp[1];
    viewMatrix.elements[6]  = -viewDirectionNormalized[1];
    viewMatrix.elements[7]  = 0.0f;
    viewMatrix.elements[8]  = right[2];
    viewMatrix.elements[9]  = newUp[2];
    viewMatrix.elements[10] = -viewDirectionNormalized[2];
    viewMatrix.elements[11] = 0.0f;
    viewMatrix.elements[12] = -cameraPosition[0] * right[0] - cameraPosition[1] * right[1] - cameraPosition[2] * right[2];
    viewMatrix.elements[13] = -cameraPosition[0] * newUp[0] - cameraPosition[1] * newUp[1] - cameraPosition[2] * newUp[2];
    viewMatrix.elements[14] = cameraPosition[0] * viewDirectionNormalized[0] + cameraPosition[1] * viewDirectionNormalized[1] + cameraPosition[2] * viewDirectionNormalized[2];
    viewMatrix.elements[15] = 1.0f;

    return viewMatrix;
}

Mat3x3 TransposeMat3x3(const Mat3x3 matrix)
{
    Mat3x3 result;
    result.elements[0] = matrix.elements[0];
    result.elements[1] = matrix.elements[3];
    result.elements[2] = matrix.elements[6];
    result.elements[3] = matrix.elements[1];
    result.elements[4] = matrix.elements[4];
    result.elements[5] = matrix.elements[7];
    result.elements[6] = matrix.elements[2];
    result.elements[7] = matrix.elements[5];
    result.elements[8] = matrix.elements[8];
    return result;
}

Mat3x3 InverseMat3x3FromMat4x4Transpose(const Mat4x4 mat4x4)
{
    Mat3x3 mat3x3 = InverseMat3x3FromMat4x4(mat4x4);
    mat3x3        = TransposeMat3x3(mat3x3);
    return mat3x3;
}

Mat3x3 Mat4x4_to_Mat3x3(const Mat4x4 m)
{
    Mat3x3 res; // create a 3x3 matrix for the model matrix
    res.elements[0] = m.elements[0];
    res.elements[1] = m.elements[1];
    res.elements[2] = m.elements[2];

    res.elements[3] = m.elements[4];
    res.elements[4] = m.elements[5];
    res.elements[5] = m.elements[6];

    res.elements[6] = m.elements[8];
    res.elements[7] = m.elements[9];
    res.elements[8] = m.elements[10];

    return res;
}

Mat3x3 Inverse_Transpose_Mat4x4_to_Mat3x3(const Mat4x4 worldMatrix)
{
    Mat3x3 modelMatrix; // create a 3x3 matrix for the model matrix
    modelMatrix.elements[0] = worldMatrix.elements[0];
    modelMatrix.elements[1] = worldMatrix.elements[1];
    modelMatrix.elements[2] = worldMatrix.elements[2];

    modelMatrix.elements[3] = worldMatrix.elements[4];
    modelMatrix.elements[4] = worldMatrix.elements[5];
    modelMatrix.elements[5] = worldMatrix.elements[6];

    modelMatrix.elements[6] = worldMatrix.elements[8];
    modelMatrix.elements[7] = worldMatrix.elements[9];
    modelMatrix.elements[8] = worldMatrix.elements[10];

    // compute the inverse transpose of the model matrix
    Mat3x3      modelMatrixInvTranspose;
    const float det = modelMatrix.elements[0] * (modelMatrix.elements[4] * modelMatrix.elements[8] - modelMatrix.elements[7] * modelMatrix.elements[5]) -
                      modelMatrix.elements[1] * (modelMatrix.elements[3] * modelMatrix.elements[8] - modelMatrix.elements[6] * modelMatrix.elements[5]) +
                      modelMatrix.elements[2] * (modelMatrix.elements[3] * modelMatrix.elements[7] - modelMatrix.elements[6] * modelMatrix.elements[4]);

    const float invDet                  = 1.0f / det;
    modelMatrixInvTranspose.elements[0] = (modelMatrix.elements[4] * modelMatrix.elements[8] - modelMatrix.elements[7] * modelMatrix.elements[5]) * invDet;
    modelMatrixInvTranspose.elements[1] = (modelMatrix.elements[2] * modelMatrix.elements[7] - modelMatrix.elements[1] * modelMatrix.elements[8]) * invDet;
    modelMatrixInvTranspose.elements[2] = (modelMatrix.elements[1] * modelMatrix.elements[5] - modelMatrix.elements[2] * modelMatrix.elements[4]) * invDet;
    modelMatrixInvTranspose.elements[3] = (modelMatrix.elements[5] * modelMatrix.elements[6] - modelMatrix.elements[3] * modelMatrix.elements[8]) * invDet;
    modelMatrixInvTranspose.elements[4] = (modelMatrix.elements[0] * modelMatrix.elements[8] - modelMatrix.elements[2] * modelMatrix.elements[6]) * invDet;
    modelMatrixInvTranspose.elements[5] = (modelMatrix.elements[2] * modelMatrix.elements[3] - modelMatrix.elements[0] * modelMatrix.elements[5]) * invDet;
    modelMatrixInvTranspose.elements[6] = (modelMatrix.elements[3] * modelMatrix.elements[7] - modelMatrix.elements[4] * modelMatrix.elements[6]) * invDet;
    modelMatrixInvTranspose.elements[7] = (modelMatrix.elements[1] * modelMatrix.elements[6] - modelMatrix.elements[0] * modelMatrix.elements[7]) * invDet;
    modelMatrixInvTranspose.elements[8] = (modelMatrix.elements[0] * modelMatrix.elements[4] - modelMatrix.elements[1] * modelMatrix.elements[3]) * invDet;

    return modelMatrixInvTranspose;
}

Mat3x3 Mul_Mat3x3(const float *A, const float *B)
{
    Mat3x3 result = {0};

    result.elements[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    result.elements[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    result.elements[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    result.elements[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    result.elements[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    result.elements[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    result.elements[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    result.elements[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    result.elements[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];

    return result;
}

Mat3x3 Create_TBN(const __m128 Tangent, const __m128 Normal)
{
    __m128 Bitangent = Vector_Cross_Product_SIMD(Normal, Tangent);

    Mat3x3 TBN;
    TBN.elements[0] = Tangent.m128_f32[0];
    TBN.elements[1] = Bitangent.m128_f32[0];
    TBN.elements[2] = Normal.m128_f32[0];
    TBN.elements[3] = Tangent.m128_f32[1];
    TBN.elements[4] = Bitangent.m128_f32[1];
    TBN.elements[5] = Normal.m128_f32[1];
    TBN.elements[6] = Tangent.m128_f32[2];
    TBN.elements[7] = Bitangent.m128_f32[2];
    TBN.elements[8] = Normal.m128_f32[2];

    return TBN;
}

Mat4x4 TransposeMat4x4(Mat4x4 *mat)
{
    Mat4x4 res = {0};

    __m128 row1 = _mm_load_ps(&mat->elements[0]);
    __m128 row2 = _mm_load_ps(&mat->elements[4]);
    __m128 row3 = _mm_load_ps(&mat->elements[8]);
    __m128 row4 = _mm_load_ps(&mat->elements[12]);

    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);

    _mm_store_ps(&res.elements[0], row1);
    _mm_store_ps(&res.elements[4], row2);
    _mm_store_ps(&res.elements[8], row3);
    _mm_store_ps(&res.elements[12], row4);

    // float temp;
    // for (int i = 0; i < 4; i++)
    //{
    //     for (int j = i + 1; j < 4; j++)
    //     {
    //         temp                    = res.elements[4 * i + j];
    //         res.elements[4 * i + j] = res.elements[4 * j + i];
    //         res.elements[4 * j + i] = temp;
    //     }
    // }

    return res;
}

Mat4x4 InverseMat4x4(Mat4x4 *mat)
{
    float *m = mat->elements;

    Mat4x4 res = {0};

    float det = 0.0f;

    res.elements[0] = m[5] * m[10] * m[15] -
                      m[5] * m[11] * m[14] -
                      m[9] * m[6] * m[15] +
                      m[9] * m[7] * m[14] +
                      m[13] * m[6] * m[11] -
                      m[13] * m[7] * m[10];

    res.elements[4] = -m[4] * m[10] * m[15] +
                      m[4] * m[11] * m[14] +
                      m[8] * m[6] * m[15] -
                      m[8] * m[7] * m[14] -
                      m[12] * m[6] * m[11] +
                      m[12] * m[7] * m[10];

    res.elements[8] = m[4] * m[9] * m[15] -
                      m[4] * m[11] * m[13] -
                      m[8] * m[5] * m[15] +
                      m[8] * m[7] * m[13] +
                      m[12] * m[5] * m[11] -
                      m[12] * m[7] * m[9];

    res.elements[12] = -m[4] * m[9] * m[14] +
                       m[4] * m[10] * m[13] +
                       m[8] * m[5] * m[14] -
                       m[8] * m[6] * m[13] -
                       m[12] * m[5] * m[10] +
                       m[12] * m[6] * m[9];

    res.elements[1] = -m[1] * m[10] * m[15] +
                      m[1] * m[11] * m[14] +
                      m[9] * m[2] * m[15] -
                      m[9] * m[3] * m[14] -
                      m[13] * m[2] * m[11] +
                      m[13] * m[3] * m[10];

    res.elements[5] = m[0] * m[10] * m[15] -
                      m[0] * m[11] * m[14] -
                      m[8] * m[2] * m[15] +
                      m[8] * m[3] * m[14] +
                      m[12] * m[2] * m[11] -
                      m[12] * m[3] * m[10];

    res.elements[9] = -m[0] * m[9] * m[15] +
                      m[0] * m[11] * m[13] +
                      m[8] * m[1] * m[15] -
                      m[8] * m[3] * m[13] -
                      m[12] * m[1] * m[11] +
                      m[12] * m[3] * m[9];

    res.elements[13] = m[0] * m[9] * m[14] -
                       m[0] * m[10] * m[13] -
                       m[8] * m[1] * m[14] +
                       m[8] * m[2] * m[13] +
                       m[12] * m[1] * m[10] -
                       m[12] * m[2] * m[9];

    res.elements[2] = m[1] * m[6] * m[15] -
                      m[1] * m[7] * m[14] -
                      m[5] * m[2] * m[15] +
                      m[5] * m[3] * m[14] +
                      m[13] * m[2] * m[7] -
                      m[13] * m[3] * m[6];

    res.elements[6] = -m[0] * m[6] * m[15] +
                      m[0] * m[7] * m[14] +
                      m[4] * m[2] * m[15] -
                      m[4] * m[3] * m[14] -
                      m[12] * m[2] * m[7] +
                      m[12] * m[3] * m[6];

    res.elements[10] = m[0] * m[5] * m[15] -
                       m[0] * m[7] * m[13] -
                       m[4] * m[1] * m[15] +
                       m[4] * m[3] * m[13] +
                       m[12] * m[1] * m[7] -
                       m[12] * m[3] * m[5];

    res.elements[14] = -m[0] * m[5] * m[14] +
                       m[0] * m[6] * m[13] +
                       m[4] * m[1] * m[14] -
                       m[4] * m[2] * m[13] -
                       m[12] * m[1] * m[6] +
                       m[12] * m[2] * m[5];

    res.elements[3] = -m[1] * m[6] * m[11] +
                      m[1] * m[7] * m[10] +
                      m[5] * m[2] * m[11] -
                      m[5] * m[3] * m[10] -
                      m[9] * m[2] * m[7] +
                      m[9] * m[3] * m[6];

    res.elements[7] = m[0] * m[6] * m[11] -
                      m[0] * m[7] * m[10] -
                      m[4] * m[2] * m[11] +
                      m[4] * m[3] * m[10] +
                      m[8] * m[2] * m[7] -
                      m[8] * m[3] * m[6];

    res.elements[11] = -m[0] * m[5] * m[11] +
                       m[0] * m[7] * m[9] +
                       m[4] * m[1] * m[11] -
                       m[4] * m[3] * m[9] -
                       m[8] * m[1] * m[7] +
                       m[8] * m[3] * m[5];

    res.elements[15] = m[0] * m[5] * m[10] -
                       m[0] * m[6] * m[9] -
                       m[4] * m[1] * m[10] +
                       m[4] * m[2] * m[9] +
                       m[8] * m[1] * m[6] -
                       m[8] * m[2] * m[5];

    det = m[0] * res.elements[0] + m[1] * res.elements[4] + m[2] * res.elements[8] + m[3] * res.elements[12];

    det = 1.0f / det;

    for (int i = 0; i < 16; i++)
        res.elements[i] *= det;

    return res;
}