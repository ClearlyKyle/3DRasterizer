#ifndef __OBJLOADER_H__
#define __OBJLOADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "./libs/tinyobj_loader_c.h"

//#include "vector.h"

#define DEBUG_PRINT 0
#define debug_pr(fmt, ...)                     \
    do                                         \
    {                                          \
        if (DEBUG_PRINT)                       \
            fprintf(stderr, fmt, __VA_ARGS__); \
    } while (0)

typedef struct Mesh_Data_s
{
    unsigned int num_of_triangles;
    float *vertex_coordinates;
    float *uv_coordinates;
} Mesh_Data;

static void loadFile(void *ctx, const char *filename, const int is_mtl, const char *obj_filename, char **buffer, size_t *len)
{
    size_t string_size = 0, read_size = 0;
    FILE *handler;
    errno_t err;

    // Open for read (will fail if file "crt_fopen_s.c" doesn't exist)
    err = fopen_s(&handler, filename, "r");
    if (err != 0)
    {
        fprintf(stderr, "[loadFile] File : %s was NOT opened!\n", filename);
        exit(1);
    }

    if (handler)
    {
        fseek(handler, 0, SEEK_END);
        string_size = ftell(handler);
        rewind(handler);
        *buffer = (char *)malloc(sizeof(char) * (string_size + 1));
        read_size = fread(*buffer, sizeof(char), (size_t)string_size, handler);
        (*buffer)[string_size] = '\0';
        if (string_size != read_size)
        {
            free(buffer);
            *buffer = NULL;
        }
        fclose(handler);
    }

    *len = read_size;
}

void Print_Attribute_Info(tinyobj_attrib_t *attrib)
{
    fprintf(stderr, "tinyobj_attrib_t ____________\n");
    fprintf(stderr, "num_vertices   \t\t: %d\n", attrib->num_vertices);
    fprintf(stderr, "num_normals    \t\t: %d\n", attrib->num_normals);
    fprintf(stderr, "num_texcoords  \t\t: %d\n", attrib->num_texcoords);
    fprintf(stderr, "num_faces      \t\t: %d\n", attrib->num_faces);
    fprintf(stderr, "num_face_num_verts\t: %d\n", attrib->num_face_num_verts);
    fprintf(stderr, "\n");
}

void Print_Material_Info(tinyobj_material_t *material)
{
    printf("\ntinyobj_material_t ____________\n");
    printf("name : %s\n", material->name);

    printf("ambient  \t\t: {%f, %f, %f}\n", material->ambient[0], material->ambient[1], material->ambient[2]);
    printf("diffuse  \t\t: {%f, %f, %f}\n", material->diffuse[0], material->diffuse[1], material->diffuse[2]);
    printf("specular \t\t: {%f, %f, %f}\n", material->specular[0], material->specular[1], material->specular[2]);
    printf("transmittance \t\t: {%f, %f, %f}\n", material->transmittance[0], material->transmittance[1], material->transmittance[2]);
    printf("emission \t\t: {%f, %f, %f}\n", material->emission[0], material->emission[1], material->emission[2]);

    printf("shininess   \t\t: %f\n", material->shininess);
    printf("ior   \t\t\t: %f\n", material->ior);
    printf("dissolve   \t\t: %f\n", material->dissolve);

    printf("map_Ka   \t\t: %s\n", material->ambient_texname);
    printf("map_Kd   \t\t: %s\n", material->diffuse_texname);
    printf("map_Ks   \t\t: %s\n", material->specular_texname);
    printf("map_Ns   \t\t: %s\n", material->specular_highlight_texname);

    printf("map_bump \t\t: %s\n", material->bump_texname);
    printf("disp     \t\t: %s\n", material->displacement_texname);
    printf("map_d    \t\t: %s\n", material->alpha_texname);
    printf("\n");
}

static void Get_Mesh_Data(const tinyobj_attrib_t *attrib, Mesh_Data *return_mesh)
{
    // Check if mesh is triangulated
    const unsigned int triangulated = (unsigned int)ceil((float)attrib->num_faces / (float)attrib->num_face_num_verts);
    fprintf(stderr, "triangulated = %d\n", triangulated);
    // fprintf(stderr, "triangulated = %f\n", (float)attrib->num_faces / (float)attrib->num_face_num_verts);
    // fprintf(stderr, "triangulated = %f\n", ceil((float)attrib->num_faces / (float)attrib->num_face_num_verts));

    return_mesh->num_of_triangles = attrib->num_faces;

    return_mesh->vertex_coordinates = (float *)malloc(sizeof(float) * (attrib->num_faces + 1) * 4);
    return_mesh->uv_coordinates = (float *)malloc(sizeof(float) * (attrib->num_faces + 1) * 2);

    if (triangulated == 3)
    {
        // 3 face values
        fprintf(stderr, "Using 3 face values setup...\n");
        for (size_t i = 0; i < attrib->num_faces; i++)
        {
            // verts
            return_mesh->vertex_coordinates[i + 0] = attrib->vertices[(int)attrib->faces[i].v_idx * 3 + 0]; // X
            return_mesh->vertex_coordinates[i + 1] = attrib->vertices[(int)attrib->faces[i].v_idx * 3 + 1]; // Y
            return_mesh->vertex_coordinates[i + 2] = attrib->vertices[(int)attrib->faces[i].v_idx * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i + 3] = 1.0f;                                                  // W

            // tex
            return_mesh->uv_coordinates[i + 0] = attrib->texcoords[(int)attrib->faces[i].vt_idx * 2 + 0]; // u
            return_mesh->uv_coordinates[i + 1] = attrib->texcoords[(int)attrib->faces[i].vt_idx * 2 + 1]; // v
        }
    }
    else if (triangulated == 4)
    {
        // 4 face values
        fprintf(stderr, "Using 4 face values setup...\n");
        for (size_t i = 0; i < attrib->num_face_num_verts - 3; i++) // loop through in steps of 4
        {
            // attrib->vertices - has the data arranged like : X Y Z X Y Z X Y Z

            const unsigned int vert1 = (unsigned int)attrib->faces[4 * i + 0].v_idx;
            const unsigned int vert2 = (unsigned int)attrib->faces[4 * i + 1].v_idx;
            const unsigned int vert3 = (unsigned int)attrib->faces[4 * i + 2].v_idx;
            const unsigned int vert4 = (unsigned int)attrib->faces[4 * i + 3].v_idx;

            // TRI 1
            return_mesh->vertex_coordinates[i * 24 + 0] = attrib->vertices[vert1 * 3 + 0]; // X
            return_mesh->vertex_coordinates[i * 24 + 1] = attrib->vertices[vert1 * 3 + 1]; // Y
            return_mesh->vertex_coordinates[i * 24 + 2] = attrib->vertices[vert1 * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i * 24 + 3] = 1.0f;                            // W

            return_mesh->vertex_coordinates[i * 24 + 4] = attrib->vertices[vert2 * 3 + 0]; // X
            return_mesh->vertex_coordinates[i * 24 + 5] = attrib->vertices[vert2 * 3 + 1]; // Y
            return_mesh->vertex_coordinates[i * 24 + 6] = attrib->vertices[vert2 * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i * 24 + 7] = 1.0f;                            // W

            return_mesh->vertex_coordinates[i * 24 + 8] = attrib->vertices[vert3 * 3 + 0];  // X
            return_mesh->vertex_coordinates[i * 24 + 9] = attrib->vertices[vert3 * 3 + 1];  // Y
            return_mesh->vertex_coordinates[i * 24 + 10] = attrib->vertices[vert3 * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i * 24 + 11] = 1.0f;                            // W

            return_mesh->uv_coordinates[i * 12 + 0] = attrib->texcoords[vert1 * 2 + 0]; // u
            return_mesh->uv_coordinates[i * 12 + 1] = attrib->texcoords[vert1 * 2 + 1]; // v

            return_mesh->uv_coordinates[i * 12 + 2] = attrib->texcoords[vert2 * 2 + 0]; // u
            return_mesh->uv_coordinates[i * 12 + 3] = attrib->texcoords[vert2 * 2 + 1]; // v

            return_mesh->uv_coordinates[i * 12 + 4] = attrib->texcoords[vert3 * 2 + 0]; // u
            return_mesh->uv_coordinates[i * 12 + 5] = attrib->texcoords[vert3 * 2 + 1]; // v

            // TRI 2
            return_mesh->vertex_coordinates[i * 24 + 12] = attrib->vertices[vert1 * 3 + 0]; // X
            return_mesh->vertex_coordinates[i * 24 + 13] = attrib->vertices[vert1 * 3 + 1]; // Y
            return_mesh->vertex_coordinates[i * 24 + 14] = attrib->vertices[vert1 * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i * 24 + 15] = 1.0f;                            // W

            return_mesh->vertex_coordinates[i * 24 + 16] = attrib->vertices[vert3 * 3 + 0]; // X
            return_mesh->vertex_coordinates[i * 24 + 17] = attrib->vertices[vert3 * 3 + 1]; // Y
            return_mesh->vertex_coordinates[i * 24 + 18] = attrib->vertices[vert3 * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i * 24 + 19] = 1.0f;                            // W

            return_mesh->vertex_coordinates[i * 24 + 20] = attrib->vertices[vert4 * 3 + 0]; // X
            return_mesh->vertex_coordinates[i * 24 + 21] = attrib->vertices[vert4 * 3 + 1]; // Y
            return_mesh->vertex_coordinates[i * 24 + 22] = attrib->vertices[vert4 * 3 + 2]; // Z
            return_mesh->vertex_coordinates[i * 24 + 23] = 1.0f;                            // W

            return_mesh->uv_coordinates[i * 12 + 6] = attrib->texcoords[vert1 * 2 + 0]; // u
            return_mesh->uv_coordinates[i * 12 + 7] = attrib->texcoords[vert1 * 2 + 1]; // v

            return_mesh->uv_coordinates[i * 12 + 8] = attrib->texcoords[vert2 * 2 + 0]; // u
            return_mesh->uv_coordinates[i * 12 + 9] = attrib->texcoords[vert2 * 2 + 1]; // v

            return_mesh->uv_coordinates[i * 12 + 10] = attrib->texcoords[vert3 * 2 + 0]; // u
            return_mesh->uv_coordinates[i * 12 + 11] = attrib->texcoords[vert3 * 2 + 1]; // v
        }
    }
    fprintf(stderr, "Vertex setup complete!\n");
}

Mesh_Data Get_Object_Data(const char *filename, bool print_info)
{
    tinyobj_shape_t *shape = NULL;
    tinyobj_material_t *material = NULL;
    tinyobj_attrib_t attrib;

    Mesh_Data mesh;

    size_t num_shapes;
    size_t num_materials;

    tinyobj_attrib_init(&attrib);

    const int ret = tinyobj_parse_obj(&attrib, &shape, &num_shapes, &material, &num_materials, filename, loadFile, NULL, 0);
    if (ret != TINYOBJ_SUCCESS)
    {
        printf("ERROR!\n");
        exit(1);
    }

    Get_Mesh_Data(&attrib, &mesh);

    if (print_info)
    {
        Print_Attribute_Info(&attrib);
        // Print_Material_Info(material);
    }

    return mesh;
}

#endif // __OBJLOADER_H__