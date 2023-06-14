#ifndef __OBJ_H__
#define __OBJ_H__

#include "textures.h"

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "./libs/tinyobj_loader_c.h"

static int _load_file(void *ctx, const char *filename, const int is_mtl, const char *obj_filename, char **buffer, size_t *len)
{
    (void)is_mtl;
    ctx          = NULL;
    obj_filename = NULL;

    FILE   *fp;
    errno_t err;

    // Open file for reading
    err = fopen_s(&fp, filename, "rb");
    if (err != 0 || fp == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        return TINYOBJ_ERROR_FILE_OPERATION;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    const long file_size = ftell(fp);
    rewind(fp);

    // Allocate buffer for file contents
    *buffer = (char *)malloc(file_size + 1);
    if (*buffer == NULL)
    {
        fprintf(stderr, "Error allocating memory for file contents\n");
        fclose(fp);
        return 1;
    }

    // Read file into buffer
    const size_t bytes_read = fread(*buffer, 1, file_size, fp);
    if (bytes_read != (size_t)file_size)
    {
        fprintf(stderr, "Error reading file %s\n", filename);
        fclose(fp);
        free(*buffer);
        *buffer = NULL;
        return 1;
    }

    // Null-terminate buffer
    (*buffer)[bytes_read] = '\0';

    // Set length and clean up
    *len = bytes_read;
    fclose(fp);

    return TINYOBJ_SUCCESS;
}

typedef struct ObjectData
{
    float *vertex_coordinates; // [x, y, z, 1.0][x, y,   z, 1.0]...
    float *normal_coordinates;
    float *uv_coordinates;

    unsigned int number_of_triangles;

    Texture_t diffuse;
    Texture_t bump;
} ObjectData_t;

static inline void _print_attribute_data(const tinyobj_attrib_t attrib)
{
    printf("Attribute data...\n");
    printf("\t num_vertices       : %d\n", attrib.num_vertices);       // Number of vertices in 'vertices' (the actual array length is num_vertices*3)
    printf("\t num_normals        : %d\n", attrib.num_normals);        // Number of vertices in 'normals' (the actual array length is num_normals*3)
    printf("\t num_texcoords      : %d\n", attrib.num_texcoords);      // Number of vertices in 'texcoords' (the actual array length is num_normals*2)
    printf("\t num_faces          : %d\n", attrib.num_faces);          // Array of faces (containing tinyobj_vertex_index_t information)
    printf("\t num_face_num_verts : %d\n", attrib.num_face_num_verts); // Total number of triangles in this object (length of face_num_verts)
}

static void _load_textures(ObjectData_t *obj, tinyobj_material_t *materials)
{
    ASSERT(obj);
    ASSERT(materials);

#if 1
    printf("Materials\n");
    printf("\t map_Ka   ambient_texname            :%s\n", materials->ambient_texname);
    printf("\t map_Kd   diffuse_texname            :%s\n", materials->diffuse_texname);
    printf("\t map_Ks   specular_texname           :%s\n", materials->specular_texname);
    printf("\t map_Ns   specular_highlight_texname :%s\n", materials->specular_highlight_texname);
    printf("\t map_bump bump_texname               :%s\n", materials->bump_texname); // Make sure this is "map_bump C:..." in the mtl file
    printf("\t disp     displacement_texname       :%s\n", materials->displacement_texname);
    printf("\t map_d    alpha_texname              :%s\n", materials->alpha_texname);
#endif

    if (materials->diffuse_texname == NULL)
        fprintf(stderr, "Error loading the diffuse_texname\n");

    if (materials->bump_texname == NULL)
        fprintf(stderr, "Error loading the bump_texname\n");

    ASSERT(materials->diffuse_texname && materials->bump_texname);

    obj->diffuse = Texture_Load(materials->diffuse_texname, true, true);
    obj->bump    = Texture_Load(materials->bump_texname, true, true);
}

ObjectData_t Object_Load(const char *file_name)
{
    ObjectData_t ret_obj = {0};

    tinyobj_attrib_t attribute;

    tinyobj_shape_t *shapes;
    size_t           number_of_shapes;

    tinyobj_material_t *materials;
    size_t              number_of_materials;

    tinyobj_attrib_init(&attribute);

    const int ret = tinyobj_parse_obj(&attribute, &shapes, &number_of_shapes, &materials, &number_of_materials, file_name, _load_file, NULL, TINYOBJ_FLAG_TRIANGULATE);
    ASSERT(ret == TINYOBJ_SUCCESS);

    const unsigned int number_of_triangles = attribute.num_face_num_verts;
    const unsigned int number_of_verticies = attribute.num_faces;

    printf("Loading Model : %s\n", file_name);
    printf("\t num materials : %zd\n", number_of_materials);
    printf("\t num triangles : %d\n", number_of_triangles);
    printf("\t num vertices  : %d\n", number_of_verticies);
    ret_obj.number_of_triangles = number_of_triangles;

    _print_attribute_data(attribute);

    _load_textures(&ret_obj, materials);

    float *vertex_coordinates = malloc(sizeof(float) * number_of_verticies * 4); // 4 coordinates per vertex [x, y, z, w]
    ASSERT(vertex_coordinates);
    ret_obj.vertex_coordinates = vertex_coordinates;

    float *normal_coordinates = malloc(sizeof(float) * number_of_verticies * 4); // 4 coordinates per normal [x, y, z, 0.0f]
    ASSERT(normal_coordinates);
    ret_obj.normal_coordinates = normal_coordinates;

    float *uv_coordinates = malloc(sizeof(float) * number_of_verticies * 2); // [u, v]
    ASSERT(uv_coordinates);
    ret_obj.uv_coordinates = uv_coordinates;

    for (unsigned int i = 0; i < number_of_triangles; i += 1)
    {
        ASSERT(attribute.face_num_verts[i] % 3 == 0); /* assume all triangle faces. */

        tinyobj_vertex_index_t idx0 = attribute.faces[i * 3 + 0];
        tinyobj_vertex_index_t idx1 = attribute.faces[i * 3 + 1];
        tinyobj_vertex_index_t idx2 = attribute.faces[i * 3 + 2];

        /* Get vertex data */
        vertex_coordinates[(i * 12) + 0] = attribute.vertices[idx0.v_idx * 3 + 0]; // X
        vertex_coordinates[(i * 12) + 1] = attribute.vertices[idx0.v_idx * 3 + 1]; // Y
        vertex_coordinates[(i * 12) + 2] = attribute.vertices[idx0.v_idx * 3 + 2]; // Z
        vertex_coordinates[(i * 12) + 3] = 1.0f;

        vertex_coordinates[(i * 12) + 4] = attribute.vertices[idx1.v_idx * 3 + 0]; // X
        vertex_coordinates[(i * 12) + 5] = attribute.vertices[idx1.v_idx * 3 + 1]; // Y
        vertex_coordinates[(i * 12) + 6] = attribute.vertices[idx1.v_idx * 3 + 2]; // Z
        vertex_coordinates[(i * 12) + 7] = 1.0f;

        vertex_coordinates[(i * 12) + 8]  = attribute.vertices[idx2.v_idx * 3 + 0]; // X
        vertex_coordinates[(i * 12) + 9]  = attribute.vertices[idx2.v_idx * 3 + 1]; // Y
        vertex_coordinates[(i * 12) + 10] = attribute.vertices[idx2.v_idx * 3 + 2]; // Z
        vertex_coordinates[(i * 12) + 11] = 1.0f;

        /* Get normal data */
        normal_coordinates[(i * 12) + 0] = attribute.normals[idx0.vn_idx * 3 + 0]; // X
        normal_coordinates[(i * 12) + 1] = attribute.normals[idx0.vn_idx * 3 + 1]; // Y
        normal_coordinates[(i * 12) + 2] = attribute.normals[idx0.vn_idx * 3 + 2]; // Z
        normal_coordinates[(i * 12) + 3] = 0.0f;

        normal_coordinates[(i * 12) + 4] = attribute.normals[idx1.vn_idx * 3 + 0]; // X
        normal_coordinates[(i * 12) + 5] = attribute.normals[idx1.vn_idx * 3 + 1]; // Y
        normal_coordinates[(i * 12) + 6] = attribute.normals[idx1.vn_idx * 3 + 2]; // Z
        normal_coordinates[(i * 12) + 7] = 0.0f;

        normal_coordinates[(i * 12) + 8]  = attribute.normals[idx2.vn_idx * 3 + 0]; // X
        normal_coordinates[(i * 12) + 9]  = attribute.normals[idx2.vn_idx * 3 + 1]; // Y
        normal_coordinates[(i * 12) + 10] = attribute.normals[idx2.vn_idx * 3 + 2]; // Z
        normal_coordinates[(i * 12) + 11] = 0.0f;

        /* Get texture data */
        // uv_coordinates = attribute.texcoords[idx0.vt_idx * 2 + 0]; // We could load them in a row
        // uv_coordinates = attribute.texcoords[idx1.vt_idx * 2 + 0]; // u u u, v v v, u u u, v v v
        // uv_coordinates = attribute.texcoords[idx2.vt_idx * 2 + 0];

        // uv_coordinates = attribute.texcoords[idx0.vt_idx * 2 + 1];
        // uv_coordinates = attribute.texcoords[idx1.vt_idx * 2 + 1];
        // uv_coordinates = attribute.texcoords[idx2.vt_idx * 2 + 1];

        uv_coordinates[i * 6 + 0] = attribute.texcoords[idx0.vt_idx * 2 + 0]; // u
        uv_coordinates[i * 6 + 1] = attribute.texcoords[idx0.vt_idx * 2 + 1]; // v

        uv_coordinates[i * 6 + 2] = attribute.texcoords[idx1.vt_idx * 2 + 0]; // u
        uv_coordinates[i * 6 + 3] = attribute.texcoords[idx1.vt_idx * 2 + 1]; // v

        uv_coordinates[i * 6 + 4] = attribute.texcoords[idx2.vt_idx * 2 + 0]; // u
        uv_coordinates[i * 6 + 5] = attribute.texcoords[idx2.vt_idx * 2 + 1]; // v
    }

    //// Side quest to get vertex index data...
    // float *vertex_data = malloc(sizeof(float) * obj.attribute.num_faces * 5); // 3 for vert, 2 for tex
    // int   *index_data  = malloc(sizeof(int) * obj.attribute.num_faces);
    // for (size_t i = 0; i < obj.attribute.num_faces; i++)
    //{
    //     tinyobj_vertex_index_t face = obj.attribute.faces[i];
    //     index_data[i]               = (int)i; // NOTE : We are not creating unique vertex data here

    //    vertex_data[i * 5 + 0] = obj.attribute.vertices[face.v_idx * 3 + 0];
    //    vertex_data[i * 5 + 1] = obj.attribute.vertices[face.v_idx * 3 + 1];
    //    vertex_data[i * 5 + 2] = obj.attribute.vertices[face.v_idx * 3 + 2];
    //    vertex_data[i * 5 + 3] = obj.attribute.texcoords[face.vt_idx * 2 + 0];
    //    vertex_data[i * 5 + 4] = obj.attribute.texcoords[face.vt_idx * 2 + 1];
    //}

    return ret_obj;
}

void Object_Destroy(ObjectData_t *obj)
{
    // Free dynamically allocated memory
    if (obj->vertex_coordinates != NULL)
    {
        free(obj->vertex_coordinates);
        obj->vertex_coordinates = NULL;
    }
    if (obj->normal_coordinates != NULL)
    {
        free(obj->normal_coordinates);
        obj->normal_coordinates = NULL;
    }
    if (obj->uv_coordinates != NULL)
    {
        free(obj->uv_coordinates);
        obj->uv_coordinates = NULL;
    }

    Texture_Destroy(&obj->diffuse);
    Texture_Destroy(&obj->bump);

    // Reset values
    obj->number_of_triangles = 0;

    *obj = (ObjectData_t){0};

    printf("Object Has been destroyed!\n");
}

#endif // __OBJ_H__