#include "lights.h"
#include "Renderer.h"

PointLight Get_Point_Light(float x, float y, float z,
                           float constant_atten, float linear_atten, float quad_atten)
{
    PointLight p = {0};

    p.position              = _mm_set_ps(0.0f, z, y, x);
    p.constant_attenuation  = constant_atten;
    p.linear_attenuation    = linear_atten;
    p.quadradic_attenuation = quad_atten;
    p.colour                = _mm_set1_ps(1.0f);
    // p.colour = _mm_set_ps(1.0f, 0.0f, 1.0f, 0.0f);

    return p;
}

__m128 Reflect_m128(const __m128 I, __m128 N)
{
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
    __m128 dotProduct = _mm_dp_ps(I, N, 0x7f);                                 // Compute dot product of I and N
    __m128 scaledN    = _mm_mul_ps(N, _mm_mul_ps(dotProduct, _mm_set1_ps(2))); // Scale N by 2 * dotProduct
    return _mm_sub_ps(I, scaledN);                                             // Subtract scaledN from I to get reflected vector
}

__m128 Get_Diffuse_Amount(const __m128 light_direction, const __m128 contact_position, const __m128 normal)
{
    const float diffuse_amount = (const float)fmax((double)Calculate_Dot_Product_SIMD(normal, light_direction), 0.0);

    // const __m128 diffuse_colour = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f);
    // const __m128 diffuse = _mm_set1_ps(diffuse_amount);
    const __m128 diffuse = _mm_set_ps(1.0f, diffuse_amount, diffuse_amount, diffuse_amount);

    return diffuse;
}

__m128 Get_Specular_Amount(const __m128 view_direction, const __m128 light_direction, const __m128 normal, const double strength, const double power_value, const Shading_Mode shading)
{
    float spec = 0.0f;
    if (shading == BLIN_PHONG)
    {
        const __m128 half_way_direction = Normalize_m128(_mm_add_ps(light_direction, view_direction));
        spec                            = (float)(pow(fmax(Calculate_Dot_Product_SIMD(normal, half_way_direction), 0.0), power_value) * strength);
    }
    else
    {
        const __m128 neg_light_direction = _mm_mul_ps(light_direction, _mm_set1_ps(-1.0f));
        const __m128 reflect_direction   = Reflect_m128(neg_light_direction, normal);
        spec                             = (float)(pow(fmax(Calculate_Dot_Product_SIMD(view_direction, reflect_direction), 0.0), power_value) * strength);
    }

    const __m128 specular = _mm_set_ps(1.0f, spec, spec, spec);

    return specular;
}

__m128 Calculate_Gourand_Shading(const __m128 vertex_position, const __m128 normal, const __m128 camera_position, const __m128 light_position, const __m128 diffuse_colour, const __m128 specular_colour)
{
    // Normalise the Noraml
    const __m128 N = Normalize_m128(normal);

    // Calculate L - direction to the light source
    const __m128 L = Normalize_m128(_mm_sub_ps(light_position, vertex_position));

    // Calculate E - view direction
    const __m128 E = Normalize_m128(_mm_sub_ps(camera_position, vertex_position));

    // Calculate R - the reflection vector
    const __m128 R = Reflect_m128(L, N);

    // Calculate Ambient Term: Hardcoded for now
    const float  ambient_strength = 0.2f;
    const __m128 Iamb             = _mm_set1_ps(ambient_strength);

    // Calculate Diffuse Term:
    const __m128 diffuse_amount = Calculate_Diffuse_Amount(L, vertex_position, N);
    const __m128 Idiff          = _mm_mul_ps(diffuse_colour, diffuse_amount); // Might need to set the Alpha here

    // Calculate Specular Term:
    const float specular_strength = 0.2f;
    const float shininess         = 32.0f;

    const float  dot_product     = Calculate_Dot_Product_SIMD(R, E);
    const float  specular_power  = powf(fmaxf(dot_product, 0.0), shininess);
    const __m128 specular_amount = _mm_set_ps(1.0f, specular_power, specular_power, specular_power);

    const __m128 Ispec = _mm_mul_ps(specular_colour, specular_amount);

    // const __m128 colour = _mm_add_ps(_mm_add_ps(Iamb, Idiff), Ispec);
    const __m128 colour = _mm_mul_ps(Clamp_m128(_mm_add_ps(_mm_add_ps(Iamb, Idiff), Ispec), 0.0f, 1.0f), _mm_set1_ps(255.0f));
    return colour;
}

// The Normals are interpolated!
__m128 Calculate_Phong_Shading(const __m128 frag_position, const __m128 frag_normal, const __m128 camera_position, const __m128 light_position, const __m128 diffuse_colour, const __m128 specular_colour)
{
    // Normalise the Noraml
    const __m128 N = Normalize_m128(frag_normal);

    // Calculate L - direction to the light source
    const __m128 L = Normalize_m128(_mm_sub_ps(light_position, frag_position));

    // Calculate E - view direction
    const __m128 E = Normalize_m128(_mm_sub_ps(camera_position, frag_position));

    // Calculate R - the reflection vector
    const __m128 R = Reflect_m128(L, N);

    // Calculate Ambient Term: Hardcoded for now
    const float  ambient_strength = 0.2f;
    const __m128 Iamb             = _mm_set1_ps(ambient_strength);

    // Calculate Diffuse Term:
    const __m128 diffuse_amount = Calculate_Diffuse_Amount(L, frag_position, N);
    const __m128 Idiff          = _mm_mul_ps(diffuse_colour, diffuse_amount); // Might need to set the Alpha here

    // Calculate Specular Term:
    const float specular_strength = 0.2f;
    const float shininess         = 32.0f;

    const float  dot_product     = Calculate_Dot_Product_SIMD(R, E);
    const float  specular_power  = powf(fmaxf(dot_product, 0.0), shininess);
    const __m128 specular_amount = _mm_set_ps(1.0f, specular_power, specular_power, specular_power);

    const __m128 Ispec = _mm_mul_ps(specular_colour, specular_amount);

    // const __m128 colour = _mm_add_ps(_mm_add_ps(Iamb, Idiff), Ispec);
    const __m128 colour = _mm_mul_ps(Clamp_m128(_mm_add_ps(_mm_add_ps(Iamb, Idiff), Ispec), 0.0f, 1.0f), _mm_set1_ps(255.0f));
    return colour;
}
