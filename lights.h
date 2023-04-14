#ifndef __LIGHTS_H__
#define __LIGHTS_H__

#include <xmmintrin.h>

typedef enum
{
    WIRE_FRAME,
    FLAT,
    GOURAND,
    PHONG,
    BLIN_PHONG,
    TEXTURED,
    TEXTURED_PHONG,
    NORMAL_MAPPING,
    SHADING_COUNT
} Shading_Mode;

typedef struct Light_s
{
    __m128 position;
    __m128 ambient_colour;
    __m128 diffuse_colour;
    __m128 specular_colour;
} Light;

// PHONG or GOURAND shading
__m128 Light_Calculate_Shading(const Shading_Mode mode, const __m128 position, const __m128 normal, const Light *light);

// typedef struct Fragment_s
//{
//     __m128 world_traingle[3];
//     __m128 surface_normal;
//     __m128 ambient, color, diffuse, specular;

//    unsigned int pixel_x;
//    unsigned int pixel_y;
//    unsigned int pixel_z;

//} Fragment;

// typedef struct PointLight_s
//{
//     __m128 position;
//     __m128 colour;

//    float constant_attenuation;
//    float linear_attenuation;
//    float quadradic_attenuation;
//} PointLight;

// PointLight Get_Point_Light(float x, float y, float z,
//                            float constant_atten, float linear_atten, float quad_atten)
//{
//     PointLight p = {0};

//    p.position              = _mm_set_ps(0.0f, z, y, x);
//    p.constant_attenuation  = constant_atten;
//    p.linear_attenuation    = linear_atten;
//    p.quadradic_attenuation = quad_atten;
//    p.colour                = _mm_set1_ps(1.0f);
//    // p.colour = _mm_set_ps(1.0f, 0.0f, 1.0f, 0.0f);

//    return p;
//}

#endif // __LIGHTS_H__