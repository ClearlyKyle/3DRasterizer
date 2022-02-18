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

__m128 Phong_Equation(const __m128 lights, const __m128 N, const __m128 V, const __m128 vertex_position, const __m128 vertex_color, const __m128 diffuse_color, const __m128 specular_color, const float shininess);
void Shade(const Fragment frag);

#endif // __LIGHTS_H__