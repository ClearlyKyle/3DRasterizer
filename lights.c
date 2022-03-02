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

    const float attenulation_value = 1.0f / (pl.constant_attenuation +
                                             pl.linear_attenuation * distance_to_light +
                                             pl.quadradic_attenuation * distance_to_light * distance_to_light);

    const float diffuse = attenulation_value * (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, direction_to_light), 0.0);
    const __m128 diffuse_colour = _mm_mul_ps(_mm_set1_ps(diffuse), _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f));

    const __m128 view_direction = Normalize_m128(_mm_sub_ps(_mm_setzero_ps(), vert));
    const __m128 spec_colour = Specular_Highlight_Colour(pl, view_direction, direction_to_light, normal);

    //// return _mm_mul_ps(pl.colour, _mm_set1_ps(dp));
    // return spec_colour;
    return _mm_add_ps(diffuse_colour, spec_colour);
}

// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
// I - 2.0 * dot(N, I) * N
__m128 Reflect_m128(const __m128 I, const __m128 N)
{
    __m128 righ = _mm_mul_ps(_mm_set1_ps(Calculate_Dot_Product_SIMD(N, I)), N);
    righ = _mm_mul_ps(_mm_set1_ps(2.0f), righ);
    return _mm_sub_ps(I, righ);
}

__m128 Specular_Highlight_Colour(const PointLight pl, const __m128 view_direction, const __m128 light_direction, const __m128 normal)
{
    // vec3 viewDir = normalize(viewPos - FragPos);
    const __m128 neg_light_direction = _mm_mul_ps(light_direction, _mm_set1_ps(-1.0f));
    const __m128 reflect_direction = Normalize_m128(Reflect_m128(neg_light_direction, normal));

    const float spec = (const float)pow(fmax(Calculate_Dot_Product_SIMD(view_direction, reflect_direction), 0.0), 128);

    const __m128 specular_strength = _mm_set1_ps(0.5f);
    const __m128 light_colour = _mm_set_ps(1.0f, 1.0f, 0.0f, 0.0f);

    return _mm_mul_ps(specular_strength, _mm_mul_ps(light_colour, _mm_set1_ps(spec)));
}