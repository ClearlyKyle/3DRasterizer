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
__m128 Light_Calculate_Shading(const __m128 position, const __m128 normal, const Light *light);

#endif // __LIGHTS_H__