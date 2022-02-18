#include "lights.h"

__m128 Phong_Equation(const __m128 lights, const __m128 N, const __m128 V, const __m128 vertex_position, const __m128 vertex_color, const __m128 diffuse_color, const __m128 specular_color, const float shininess)
{
    // const __m128 ambient(0.1);
    __m128 diffuse;
    __m128 specular;

    // for (size_t i = 0; i < num_of_lights; i++)
    //{
    // }
    const Vector3D L = -(light->getDirectionToPoint(vertex_position));
    const Vector3D R = 2 * (N * L) * N - L;

    diffuse = light->getColor() * diffuse_color * std::max((L * N), 0.0);
    specular = light->getColor() * specular_color * pow(std::max((R * V), 0.0), shininess);

    const RGBColor phong_result = (ambient + diffuse + specular) * vertex_color;
    return phong_result;
}

void Shade(const Fragment frag)
{
}

// Raster/screen space to NDC [-1,1]
static void viewportTransformInv(const __m128 x, const __m128 y, __m128 *x_ndc, __m128 *y_ndc)
{
    const float slopeX = 2.0 / 900;
    const float slopeY = 2.0 / 1000;

    *x_ndc = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(slopeX)), _mm_set1_ps(-1.0f));
    *y_ndc = _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(slopeY)), _mm_set1_ps(-1.0f));
}

// NDC [-1,1] to Camera space
__m128 projectTransformInv(const Point2D &point_projected, const double depth)
{
    const __m128 point = _mm_set_ps(1.0f, depth, point_projected.y, point_projected.x);
    const __m128 r = Matrix_Multiply_Vector_SIMD(m_project_inv, point);

    return Point3D(r.x, r.y, depth);
}

// Camera/view space to World space
const Point3D Camera::viewTransformInv(const Point3D &point_camera) const
{
    glm::vec4 p = glm::vec4(point_camera.x, point_camera.y, point_camera.z, 1);
    glm::vec4 r = p * m_lookat_inv;

    return Point3D(r.x, r.y, r.z);
    ;
}