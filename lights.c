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

__m128 Calculate_Point_Light_Colour(const PointLight pl, __m128 normal, __m128 frag_position)
{
    //// calculate vertex to light data
    // const __m128 vert_to_light = _mm_sub_ps(pl.position, vert);

    //// calculate distance
    // vert.m128_f32[3] = 0.0f;

    // const float distance_to_light = (const float)sqrt((double)hsum_ps_sse3(_mm_mul_ps(vert_to_light, vert_to_light)));

    //// calculate direction
    // const __m128 direction_to_light = Normalize_m128(_mm_sub_ps(vert_to_light, _mm_set1_ps(distance_to_light)));

    // const float attenulation_value = 1.0f / (pl.constant_attenuation +
    //                                          pl.linear_attenuation * distance_to_light +
    //                                          pl.quadradic_attenuation * distance_to_light * distance_to_light);

    const float specular_strength = 0.8f;
    const float ambient_value = 0.1f;
    normal = Normalize_m128(normal);

    // AMBIENT
    const __m128 ambient = _mm_set1_ps(ambient_value);

    const __m128 light_position = _mm_set_ps(0.0f, 2.0f, 0.0f, -1.0f);
    const __m128 light_direction = Normalize_m128(_mm_sub_ps(light_position, frag_position));
    const __m128 negative_light_direction = _mm_mul_ps(light_direction, _mm_set1_ps(-1.0f));

    // DIFFUSE
    const float diffuse_amount = (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, light_direction), 0.0);
    // const float diffuse = attenulation_value * (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, direction_to_light), 0.0);
    const __m128 diffuse_colour = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f);
    const __m128 diffuse = _mm_mul_ps(_mm_set1_ps(diffuse_amount), diffuse_colour);

    // SPECULAR
    const __m128 camera_position = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    const __m128 view_direction = Normalize_m128(_mm_sub_ps(camera_position, frag_position));
    const __m128 half_way_direction = Normalize_m128(_mm_add_ps(light_direction, view_direction));

    const __m128 reflection_direction = Normalize_m128(Reflect_m128(negative_light_direction, normal));

    // const float spec_amount = (const float)pow(fmax(Calculate_Dot_Product_SIMD(view_direction, reflection_direction), 0.0), 32);
    const float spec_amount = (const float)pow(fmax(Calculate_Dot_Product_SIMD(normal, half_way_direction), 0.0), 16);

    const __m128 specular_colour = _mm_set_ps(1.0f, 1.0f, 0.0f, 0.0f);
    const __m128 specular = _mm_mul_ps(specular_colour, _mm_set1_ps(spec_amount * specular_strength));

    const __m128 final_colour = _mm_add_ps(_mm_add_ps(ambient, diffuse), specular);

    return final_colour;
}

// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
// I - 2.0 * dot(N, I) * N
__m128 Reflect_m128(const __m128 I, __m128 N)
{
    N = Normalize_m128(N);
    __m128 righ = _mm_mul_ps(_mm_set1_ps(Calculate_Dot_Product_SIMD(N, I)), N);
    righ = _mm_mul_ps(righ, _mm_set1_ps(2.0f));
    return _mm_sub_ps(I, righ);
}

__m128 Specular_Highlight_Colour(const __m128 view_direction, const __m128 light_direction, const __m128 normal)
{
    // vec3 viewDir = normalize(viewPos - FragPos);
    const __m128 neg_light_direction = _mm_mul_ps(light_direction, _mm_set1_ps(-1.0f));
    const __m128 reflect_direction = Reflect_m128(neg_light_direction, normal);

    const float spec = (const float)pow(fmax(Calculate_Dot_Product_SIMD(view_direction, reflect_direction), 0.0), 16);

    const __m128 specular_strength = _mm_set1_ps(0.5f);
    const __m128 light_colour = _mm_set_ps(1.0f, 1.0f, 0.0f, 0.0f);

    return _mm_mul_ps(specular_strength, _mm_mul_ps(light_colour, _mm_set1_ps(spec)));
}

static double clampf(double d, double min, double max)
{
    const double t = d < min ? min : d;
    return t > max ? max : t;
}

__m128 Calculate_Normal_Mapping_Colour(const unsigned char *diffuse_texture,
                                       const unsigned char *normal_texture,
                                       const Mat4x4 TBN,
                                       const __m128 Tangent_Frag_Pos,
                                       const __m128 Tangent_Light_Pos,
                                       const __m128 Tangent_View_Pos)
{
    const float specular_strength = 0.2f;
    const float ambient_value = 0.1f;

    // AMBIENT
    const __m128 ambient = _mm_set1_ps(ambient_value);

    // COLOUR FROM TEXTURE
    //__m128i colour_test = _mm_load_si128((__m128i *)diffuse_texture);
    __m128 texture_colour = _mm_set_ps(1.0f, diffuse_texture[2] / 255.0f, diffuse_texture[1] / 255.0f, diffuse_texture[0] / 255.0f);

    // NORMAL FROM TEXTURE : transform normal vector to range [-1,1]
    //__m128 normal = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    __m128 normal = _mm_set_ps(1.0f, normal_texture[2] / 256.0f, normal_texture[1] / 256.0f, normal_texture[0] / 256.0f);
    normal = _mm_sub_ps(_mm_mul_ps(normal, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));

    const __m128 light_direction = Normalize_m128(_mm_sub_ps(Tangent_Light_Pos, Tangent_Frag_Pos));
    const __m128 negative_light_direction = _mm_mul_ps(light_direction, _mm_set1_ps(-1.0f));

    // DIFFUSE
    const float diffuse_amount = (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, light_direction), 0.0);
    //const __m128 diffuse_colour = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f);
    const __m128 diffuse = _mm_set1_ps(diffuse_amount);
    // const __m128 diffuse = _mm_mul_ps(_mm_set1_ps(diffuse_amount), diffuse_colour);

    // SPECULAR
    const __m128 view_direction = Normalize_m128(_mm_sub_ps(Tangent_View_Pos, Tangent_Frag_Pos));
    // const __m128 half_way_direction = Normalize_m128(_mm_add_ps(light_direction, view_direction));

    const __m128 reflection_direction = Normalize_m128(Reflect_m128(negative_light_direction, normal));

    // const float spec = (const float)pow(fmax(Calculate_Dot_Product_SIMD(normal, halfwayDir), 0.0), 16);
    const float spec = (const float)pow(fmax(Calculate_Dot_Product_SIMD(view_direction, reflection_direction), 0.0), 16);

    //const __m128 specular_colour = _mm_set_ps(1.0f, 0.0f, 1.0f, 0.0f);
    // const __m128 specular = _mm_mul_ps(_mm_set1_ps(spec * specular_strength), specular_colour);
    const __m128 specular = _mm_set1_ps(spec * specular_strength);

    // FINAL COLOUR
    const __m128 final_colour = _mm_add_ps(_mm_mul_ps(_mm_add_ps(ambient, diffuse), texture_colour), specular);

    return final_colour;
}