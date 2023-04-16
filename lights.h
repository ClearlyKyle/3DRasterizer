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

static const char *Shading_Mode_Str[SHADING_COUNT] = {
    "WIRE_FRAME",
    "FLAT",
    "GOURAND",
    "PHONG",
    "BLIN_PHONG",
    "TEXTURED",
    "TEXTURED_PHONG",
    "NORMAL_MAPPING",
};

typedef struct Light_s
{
    __m128 position;
    __m128 ambient_colour;
    __m128 diffuse_colour;
    __m128 specular_colour;
} Light;

__m128 Light_Calculate_Shading(const __m128 position, const __m128 normal, const __m128 camera_position, const __m128 light_position, const Light *light);

#endif // __LIGHTS_H__