#ifndef __MAT3_H__
#define __MAT3_H__

#include "common.h"
#include "vec4.h"

typedef struct
{
    float f[3][3];
} mmat3;

// mmat3 mate_mat3_transpose(const mmat3 m)
//{
//     mmat3 res = {0};
//     res.f[0]  = m.f[0];
//     res.f[1]  = m.f[3];
//     res.f[2]  = m.f[6];
//     res.f[3]  = m.f[1];
//     res.f[4]  = m.f[4];
//     res.f[5]  = m.f[7];
//     res.f[6]  = m.f[2];
//     res.f[7]  = m.f[5];
//     res.f[8]  = m.f[8];
//     return res;
// }

// mmat3 mate_tbn_create(const mvec4 Tangent, const mvec4 Normal)
//{
//     mvec4 Bitangent = mate_cross(Normal, Tangent);

//    mmat3 TBN       = {0};
//    TBN.elements[0] = Tangent.m128_f32[0];
//    TBN.elements[1] = Bitangent.m128_f32[0];
//    TBN.elements[2] = Normal.m128_f32[0];
//    TBN.elements[3] = Tangent.m128_f32[1];
//    TBN.elements[4] = Bitangent.m128_f32[1];
//    TBN.elements[5] = Normal.m128_f32[1];
//    TBN.elements[6] = Tangent.m128_f32[2];
//    TBN.elements[7] = Bitangent.m128_f32[2];
//    TBN.elements[8] = Normal.m128_f32[2];

//    return TBN;
//}

#endif // __MAT3_H__