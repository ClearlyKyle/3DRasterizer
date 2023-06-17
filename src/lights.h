#ifndef __LIGHTS_H__
#define __LIGHTS_H__

#include "matematika.h"

typedef enum
{
    SHADING_WIRE_FRAME,
    SHADING_FLAT,
    SHADING_GOURAUD,
    SHADING_PHONG,
    SHADING_BLIN_PHONG,
    SHADING_TEXTURED,
    SHADING_TEXTURED_PHONG,
    SHADING_NORMAL_MAPPING,
    SHADING_DEPTH_BUFFER,
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
    "DEPTH_BUFFER",
};

typedef struct Light_s
{
    mvec4 position;
    mvec4 diffuse_colour;
    mvec4 ambient_amount;
    mvec4 specular_amount;
} Light;

mvec4 Light_Calculate_Shading(const mvec4 position, const mvec4 normal, const mvec4 camera_position, const Light *light);

#endif // __LIGHTS_H__