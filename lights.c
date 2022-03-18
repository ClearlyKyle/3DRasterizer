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

__m128 Reflect_m128(const __m128 I, __m128 N)
{
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
    // I - 2.0 * dot(N, I) * N

    N = Normalize_m128(N);

    __m128 righ = _mm_mul_ps(_mm_set1_ps(Calculate_Dot_Product_SIMD(N, I)), N);
    righ = _mm_mul_ps(righ, _mm_set1_ps(2.0f));

    return _mm_sub_ps(I, righ);
}

__m128 Calculate_Light(const __m128 light_position, const __m128 camera_position, const __m128 frag_position, const __m128 normal,
                       const float ambient_strength, const float diffuse_strength, const float specular_strength, const double specular_power)
{
    // POINT LIGHT
    // const float distance_to_light = (const float)sqrt((double)hsum_ps_sse3(_mm_mul_ps(vert_to_light, vert_to_light)));
    // const float attenulation_value = 1.0f / (pl.constant_attenuation +
    //                                          pl.linear_attenuation * distance_to_light +
    //                                          pl.quadradic_attenuation * distance_to_light * distance_to_light);

    const __m128 view_direction = Normalize_m128(_mm_sub_ps(camera_position, frag_position));
    const __m128 light_direction = Normalize_m128(_mm_sub_ps(light_position, frag_position));

    const __m128 diffuse = Get_Diffuse_Amount(light_direction, frag_position, normal);
    const __m128 specular = Get_Specular_Amount(view_direction, light_direction, normal, 0.2, 64);
    const __m128 ambient = _mm_set1_ps(ambient_strength);

    const __m128 colour = _mm_mul_ps(Clamp_m128(_mm_add_ps(_mm_add_ps(ambient, diffuse), specular), 0.0f, 1.0f), _mm_set1_ps(255.0f));

    return colour;
}

__m128 Get_Diffuse_Amount(const __m128 light_direction, const __m128 contact_position, const __m128 normal)
{
    const float diffuse_amount = (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, light_direction), 0.0);

    // const __m128 diffuse_colour = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f);
    // const __m128 diffuse = _mm_set1_ps(diffuse_amount);
    const __m128 diffuse = _mm_set_ps(1.0f, diffuse_amount, diffuse_amount, diffuse_amount);

    return diffuse;
}

__m128 Get_Specular_Amount(const __m128 view_direction, const __m128 light_direction, const __m128 normal, const double strength, const double power_value)
{
    const bool blinn = false;

    float spec = 0.0f;
    if (blinn)
    {
        const __m128 half_way_direction = Normalize_m128(_mm_add_ps(light_direction, view_direction));
        spec = (const float)pow(fmax(Calculate_Dot_Product_SIMD(normal, half_way_direction), 0.0), power_value) * strength;
    }
    else
    {
        const __m128 neg_light_direction = _mm_mul_ps(light_direction, _mm_set1_ps(-1.0f));
        const __m128 reflect_direction = Reflect_m128(neg_light_direction, normal);
        spec = (const float)pow(fmax(Calculate_Dot_Product_SIMD(view_direction, reflect_direction), 0.0), power_value) * strength;
    }

    const __m128 specular = _mm_set_ps(1.0f, spec, spec, spec);

    return specular;
}