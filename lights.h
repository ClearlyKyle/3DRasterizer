#ifndef __LIGHTS_H__
#define __LIGHTS_H__

#include <xmmintrin.h>

#include "vector.h"

typedef struct Fragment_s
{
    __m128 world_traingle[3];
    __m128 surface_normal;
    __m128 ambient, color, diffuse, specular;

    unsigned int pixel_x;
    unsigned int pixel_y;
    unsigned int pixel_z;

} Fragment;

typedef struct PointLight_s
{
    __m128 position;
    __m128 colour;

    float constant_attenuation;
    float linear_attenuation;
    float quadradic_attenuation;
} PointLight;

PointLight Get_Point_Light(float x, float y, float z, float constant_atten, float linear_atten, float quad_atten);

__m128 Reflect_m128(const __m128 I, const __m128 N);

__m128 Get_Diffuse_Amount(const __m128 light_direction, const __m128 contact_position, const __m128 normal);
__m128 Get_Specular_Amount(const __m128 view_direction, const __m128 light_direction, const __m128 normal, const double strength, const double power_value);

__m128 Calculate_Light(const __m128 light_position, const __m128 camera_position, const __m128 frag_position, const __m128 normal,
                       const float ambient_strength, const float diffuse_strength, const float specular_strength, const double specular_power);

#endif // __LIGHTS_H__