#include <stdio.h>
#include <stdlib.h>

#include "vector.h"

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 1000

int main(int argc, char const *argv[])
{
    Mat4x4 World_Matrix = {0.0f};
    float fTheta = 0.0f;

    const Mat4x4 Projection_matrix = Get_Projection_Matrix(90.0f, SCREEN_HEIGHT / SCREEN_WIDTH, 0.1f, 1000.0f);
    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.0f, 5.0f);
    const Mat4x4 matRotZ = Get_Rotation_Z_Matrix(fTheta); // Rotation Z
    const Mat4x4 matRotX = Get_Rotation_X_Matrix(fTheta); // Rotation X

    Matrix_Multiply_Matrix(matRotZ.elements, matRotX.elements, World_Matrix.elements);
    Matrix_Multiply_Matrix(World_Matrix.elements, Translation_matrix.elements, World_Matrix.elements);

    // (W Z Y X)
    __m128 tri1 = _mm_set_ps(1.0f, -1.0f, 1.0f, 1.0f);
    __m128 tri2 = _mm_set_ps(1.0f, -1.0f, -1.0f, 1.0f);
    __m128 tri3 = _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f);

    tri1 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri1);
    tri2 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri2);
    tri3 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri3);

    // This works as expecetd!
    // tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));
    // tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));
    // tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));

    // Vector Dot Product between : Surface normal and CameraRay
    const __m128 camera = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

    const __m128 surface_normal = Calculate_Surface_Normal_SIMD(tri1, tri2, tri3);
    const __m128 camera_ray = _mm_sub_ps(tri1, camera);

    const float dot_product_result = Calculate_Dot_Product_SIMD(surface_normal, camera_ray);

    tri1 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri1);
    tri2 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri2);
    tri3 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri3);

    // Perform x/w, y/w, z/w
    tri1 = _mm_div_ps(tri1, _mm_shuffle_ps(tri1, tri1, _MM_SHUFFLE(3, 3, 3, 3)));
    tri2 = _mm_div_ps(tri2, _mm_shuffle_ps(tri2, tri2, _MM_SHUFFLE(3, 3, 3, 3)));
    tri3 = _mm_div_ps(tri3, _mm_shuffle_ps(tri3, tri3, _MM_SHUFFLE(3, 3, 3, 3)));

    // Sacle Into View
    tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
    tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
    tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));

    const float x_adjustment = 0.5f * SCREEN_WIDTH;
    const float y_adjustment = 0.5f * SCREEN_HEIGHT;

    tri1 = _mm_mul_ps(tri1, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
    tri2 = _mm_mul_ps(tri2, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
    tri3 = _mm_mul_ps(tri3, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
    return 0;
}
