#ifndef __LIGHTS_H__
#define __LIGHTS_H__

#include "matematika.h"

// TODO : Prefix these
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
    "SHADING_DEPTH_BUFFER",
};

typedef struct Light_s
{
    mvec4 position;
    mvec4 diffuse_colour;
    mvec4 ambient_amount;
    mvec4 specular_amount;
} Light;

mvec4 Light_Calculate_Shading(const mvec4 position, const mvec4 normal, const mvec4 camera_position, const mvec4 light_position, const Light *light);

#endif // __LIGHTS_H__