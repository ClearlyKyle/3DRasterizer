#ifndef __OBJ_H__
#define __OBJ_H__

#include "matematika.h"
#include "textures.h"

typedef struct ObjectData
{
    float *vertex_coordinates; // [x, y, z, 1.0][x, y,   z, 1.0]...
    float *normal_coordinates;
    float *uv_coordinates;

    unsigned int number_of_triangles;

    int has_textures;
    int has_normals;
    int has_texcoords;

    mmat4 transform;

    Texture_t diffuse;
    Texture_t bump;
} ObjectData_t;

ObjectData_t Object_Load(const char *file_name);
void         Object_Destroy(ObjectData_t *obj);

#endif // __OBJ_H__