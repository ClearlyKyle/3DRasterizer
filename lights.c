#include "lights.h"

PointLight Get_Point_Light(float x, float y, float z,
                           float linear_atten, float quad_atten, float constant_atten)
{
    PointLight p = {0};

    p.position = _mm_set_ps(0.0f, z, y, x);
    p.constant_attenuation = constant_atten;
    p.linear_attenuation = linear_atten;
    p.quadradic_attenuation = linear_atten;

    return p;
}