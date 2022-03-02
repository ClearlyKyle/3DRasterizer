#include "lights.h"

PointLight Get_Point_Light(float x, float y, float z,
                           float constant_atten, float linear_atten, float quad_atten)
{
    PointLight p = {0};

    p.position = _mm_set_ps(0.0f, z, y, x);
    p.constant_attenuation = constant_atten;
    p.linear_attenuation = linear_atten;
    p.quadradic_attenuation = quad_atten;
    p.colour = _mm_set1_ps(1.0f);
    // p.colour = _mm_set_ps(1.0f, 0.0f, 1.0f, 0.0f);

    return p;
}

__m128 Calculate_Point_Light_Colour(const PointLight pl, const __m128 normal, __m128 vert)
{
    // calculate vertex to light data
    const __m128 vert_to_light = _mm_sub_ps(pl.position, vert);

    // calculate distance
    vert.m128_f32[3] = 0.0f;

    const float distance_to_light = (const float)sqrt((double)hsum_ps_sse3(_mm_mul_ps(vert_to_light, vert_to_light)));

    // calculate direction
    const __m128 direction_to_light = _mm_div_ps(vert_to_light, _mm_set1_ps(distance_to_light));

    const float attenulation_value1 = 1.0f / (pl.constant_attenuation +
                                              pl.linear_attenuation * distance_to_light +
                                              pl.quadradic_attenuation * distance_to_light * distance_to_light);

    const float dp = (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, direction_to_light), 0.0);

    // return _mm_mul_ps(pl.colour, _mm_set1_ps(dp));
    return _mm_set1_ps(dp);
}

// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
// I - 2.0 * dot(N, I) * N
__m128 Reflect_m128(const __m128 I, const __m128 N)
{
    __m128 righ = _mm_mul_ps(_mm_set1_ps(Calculate_Dot_Product_SIMD(N, I)), N);
    righ = _mm_mul_ps(_mm_set1_ps(2.0f), righ);
    return _mm_mul_ps(I, righ);
}