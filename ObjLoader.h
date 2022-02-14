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
    unsigned int num_of_verticies;

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
    fprintf(stderr, "num_face_num_verts\t: %d (number of 'f' rows)\n", attrib->num_face_num_verts);
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

static void Get_Mesh_Data(const tinyobj_attrib_t *attrib, Mesh_Data **return_mesh)
{
    // Check if mesh is triangulated
    const unsigned int triangulated = (unsigned int)ceil((float)attrib->num_faces / (float)attrib->num_face_num_verts);
    fprintf(stderr, "triangulated = %d\n", triangulated);
    // fprintf(stderr, "triangulated = %f\n", (float)attrib->num_faces / (float)attrib->num_face_num_verts);
    // fprintf(stderr, "triangulated = %f\n", ceil((float)attrib->num_faces / (float)attrib->num_face_num_verts));

    float *vert_coords = NULL;
    float *uv_coords = NULL;

    if (triangulated == 3)
    {
        // 3 face values
        fprintf(stderr, "Using 3 face values setup...\n");
        const unsigned int number_of_triangles = attrib->num_face_num_verts;

        (*return_mesh)->num_of_triangles = number_of_triangles;
        (*return_mesh)->num_of_verticies = number_of_triangles * 3;

        vert_coords = (float *)malloc(sizeof(float) * number_of_triangles * 4);
        if (!vert_coords)
        {
            fprintf(stderr, "Error alocating memeory for 'vert_coords'\n");
            exit(1);
        }
        uv_coords = (float *)malloc(sizeof(float) * number_of_triangles * 2);
        if (!uv_coords)
        {
            fprintf(stderr, "Error alocating memeory for 'vert_coords'\n");
            exit(1);
        }

        for (size_t i = 0; i < attrib->num_face_num_verts; i++)
        {
            // verts
            vert_coords[i * 4 + 0] = attrib->vertices[(int)attrib->faces[i].v_idx * 3 + 0]; // X
            vert_coords[i * 4 + 1] = attrib->vertices[(int)attrib->faces[i].v_idx * 3 + 1]; // Y
            vert_coords[i * 4 + 2] = attrib->vertices[(int)attrib->faces[i].v_idx * 3 + 2]; // Z
            vert_coords[i * 4 + 3] = 1.0f;                                                  // W

            // tex
            uv_coords[i * 2 + 0] = attrib->texcoords[(int)attrib->faces[i].vt_idx * 2 + 0]; // u
            uv_coords[i * 2 + 1] = attrib->texcoords[(int)attrib->faces[i].vt_idx * 2 + 1]; // v
        }
    }
    else if (triangulated == 4)
    {
        fprintf(stderr, "Using 4 face values setup...\n");
        const unsigned int number_of_triangles = attrib->num_face_num_verts * 2;

        (*return_mesh)->num_of_triangles = number_of_triangles;
        (*return_mesh)->num_of_verticies = number_of_triangles * 3;

        vert_coords = (float *)malloc(sizeof(float) * (number_of_triangles * 3) * 4);
        if (!vert_coords)
        {
            fprintf(stderr, "Error alocating memeory for 'vert_coords'\n");
            exit(1);
        }
        uv_coords = (float *)malloc(sizeof(float) * (number_of_triangles * 3) * 2);
        if (!uv_coords)
        {
            fprintf(stderr, "Error alocating memeory for 'vert_coords'\n");
            exit(1);
        }

        // 4 face values
        for (size_t i = 0; i < attrib->num_face_num_verts; i++) // loop through in steps of 4
        {
            const int f_vert_index1 = attrib->faces[4 * i + 0].v_idx;
            const int f_vert_index2 = attrib->faces[4 * i + 1].v_idx;
            const int f_vert_index3 = attrib->faces[4 * i + 2].v_idx;
            const int f_vert_index4 = attrib->faces[4 * i + 3].v_idx;

            // TRI 1
            // attrib->vertices - has the data arranged like : X Y Z X Y Z X Y Z
            vert_coords[(i * 24) + 0] = attrib->vertices[f_vert_index1 * 3 + 0]; // X
            vert_coords[(i * 24) + 1] = attrib->vertices[f_vert_index1 * 3 + 1]; // Y
            vert_coords[(i * 24) + 2] = attrib->vertices[f_vert_index1 * 3 + 2]; // Z
            vert_coords[(i * 24) + 3] = 1.0f;                                    // W

            vert_coords[(i * 24) + 4] = attrib->vertices[f_vert_index2 * 3 + 0]; // X
            vert_coords[(i * 24) + 5] = attrib->vertices[f_vert_index2 * 3 + 1]; // Y
            vert_coords[(i * 24) + 6] = attrib->vertices[f_vert_index2 * 3 + 2]; // Z
            vert_coords[(i * 24) + 7] = 1.0f;                                    // W

            vert_coords[(i * 24) + 8] = attrib->vertices[f_vert_index3 * 3 + 0];  // X
            vert_coords[(i * 24) + 9] = attrib->vertices[f_vert_index3 * 3 + 1];  // Y
            vert_coords[(i * 24) + 10] = attrib->vertices[f_vert_index3 * 3 + 2]; // Z
            vert_coords[(i * 24) + 11] = 1.0f;                                    // W

            // TRI 2
            vert_coords[(i * 24) + 12] = attrib->vertices[f_vert_index1 * 3 + 0]; // X
            vert_coords[(i * 24) + 13] = attrib->vertices[f_vert_index1 * 3 + 1]; // Y
            vert_coords[(i * 24) + 14] = attrib->vertices[f_vert_index1 * 3 + 2]; // Z
            vert_coords[(i * 24) + 15] = 1.0f;                                    // W

            vert_coords[(i * 24) + 16] = attrib->vertices[f_vert_index3 * 3 + 0]; // X
            vert_coords[(i * 24) + 17] = attrib->vertices[f_vert_index3 * 3 + 1]; // Y
            vert_coords[(i * 24) + 18] = attrib->vertices[f_vert_index3 * 3 + 2]; // Z
            vert_coords[(i * 24) + 19] = 1.0f;                                    // W

            vert_coords[(i * 24) + 20] = attrib->vertices[f_vert_index4 * 3 + 0]; // X
            vert_coords[(i * 24) + 21] = attrib->vertices[f_vert_index4 * 3 + 1]; // Y
            vert_coords[(i * 24) + 22] = attrib->vertices[f_vert_index4 * 3 + 2]; // Z
            vert_coords[(i * 24) + 23] = 1.0f;                                    // W
        }
        for (size_t i = 0; i < attrib->num_face_num_verts; i++) // loop through in steps of 4
        {
            const int f_tex_index1 = attrib->faces[4 * i + 0].vt_idx;
            const int f_tex_index2 = attrib->faces[4 * i + 1].vt_idx;
            const int f_tex_index3 = attrib->faces[4 * i + 2].vt_idx;
            const int f_tex_index4 = attrib->faces[4 * i + 3].vt_idx;

            uv_coords[i * 12 + 0] = attrib->texcoords[f_tex_index1 * 2 + 0]; // u
            uv_coords[i * 12 + 1] = attrib->texcoords[f_tex_index1 * 2 + 1]; // v

            uv_coords[i * 12 + 2] = attrib->texcoords[f_tex_index2 * 2 + 0]; // u
            uv_coords[i * 12 + 3] = attrib->texcoords[f_tex_index2 * 2 + 1]; // v

            uv_coords[i * 12 + 4] = attrib->texcoords[f_tex_index3 * 2 + 0]; // u
            uv_coords[i * 12 + 5] = attrib->texcoords[f_tex_index3 * 2 + 1]; // v

            uv_coords[i * 12 + 6] = attrib->texcoords[f_tex_index1 * 2 + 0]; // u
            uv_coords[i * 12 + 7] = attrib->texcoords[f_tex_index1 * 2 + 1]; // v

            uv_coords[i * 12 + 8] = attrib->texcoords[f_tex_index3 * 2 + 0]; // u
            uv_coords[i * 12 + 9] = attrib->texcoords[f_tex_index3 * 2 + 1]; // v

            uv_coords[i * 12 + 10] = attrib->texcoords[f_tex_index4 * 2 + 0]; // u
            uv_coords[i * 12 + 11] = attrib->texcoords[f_tex_index4 * 2 + 1]; // v
        }
    }

    (*return_mesh)->vertex_coordinates = vert_coords;
    (*return_mesh)->uv_coordinates = uv_coords;

    fprintf(stderr, "Vertex setup complete!\n");
}

void Get_Object_Data(const char *filename, bool print_info, Mesh_Data **output)
{
    tinyobj_shape_t *shape = NULL;
    tinyobj_material_t *material = NULL;
    tinyobj_attrib_t attrib;

    size_t num_shapes;
    size_t num_materials;

    tinyobj_attrib_init(&attrib);

    const int ret = tinyobj_parse_obj(&attrib, &shape, &num_shapes, &material, &num_materials, filename, loadFile, NULL, 0);
    if (ret != TINYOBJ_SUCCESS)
    {
        fprintf(stderr, "ERROR!\n");
        exit(1);
    }

    (*output) = (Mesh_Data *)malloc(sizeof(Mesh_Data) * 1);
    Get_Mesh_Data(&attrib, output);

    if (print_info)
    {
        Print_Attribute_Info(&attrib);
        // Print_Material_Info(material);

        fprintf(stderr, "Number of Triangles = %d\n", (*output)->num_of_triangles);
        fprintf(stderr, "Number of Verticies = %d\n", (*output)->num_of_verticies);
    }
    fprintf(stderr, "Mesh complete!\n");
}

void Free_Mesh(Mesh_Data **m)
{
    // free((*m)->vertex_coordinates);
    // free((*m)->uv_coordinates);

    fprintf(stderr, "Free Mesh Sucess\n");
}

#endif // __OBJLOADER_H__