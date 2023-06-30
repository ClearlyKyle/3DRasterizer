#include "obj.h"

#define FLT_MAX 3.402823466e+38F
#define FLT_MIN 1.175494351e-38F

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"

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

    obj->diffuse = Texture_Load(materials->diffuse_texname, true);
    obj->bump    = Texture_Load(materials->bump_texname, true);
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

    if (materials)
        _load_textures(&ret_obj, materials);
    else
        printf("This model has no materials\n");

    ret_obj.has_normals   = attribute.num_normals ? 1 : 0;
    ret_obj.has_texcoords = attribute.num_texcoords ? 1 : 0;
    ret_obj.has_textures  = materials ? 1 : 0;

    float *vertex_coordinates = malloc(sizeof(float) * number_of_verticies * 4); // 4 coordinates per vertex [x, y, z, w]
    ASSERT(vertex_coordinates);
    ret_obj.vertex_coordinates = vertex_coordinates;

    float *normal_coordinates = malloc(sizeof(float) * number_of_verticies * 4); // 4 coordinates per normal [x, y, z, 0.0f]
    ASSERT(normal_coordinates);
    ret_obj.normal_coordinates = normal_coordinates;

    float *uv_coordinates = malloc(sizeof(float) * number_of_verticies * 2); // [u, v]
    ASSERT(uv_coordinates);
    ret_obj.uv_coordinates = uv_coordinates;

    float bbmin[3], bbmax[3]; // Bounding Box (x, y, z)
    bbmin[0] = bbmin[1] = bbmin[2] = FLT_MAX;
    bbmax[0] = bbmax[1] = bbmax[2] = -FLT_MAX;

    for (unsigned int i = 0; i < number_of_verticies; ++i)
    {
        ASSERT(attribute.face_num_verts[i % 3] % 3 == 0); /* assume all triangle faces. */

        tinyobj_vertex_index_t idx = attribute.faces[i];

        /* Get vertex data */
        vertex_coordinates[(i * 4) + 0] = attribute.vertices[idx.v_idx * 3 + 0]; // X
        vertex_coordinates[(i * 4) + 1] = attribute.vertices[idx.v_idx * 3 + 1]; // Y
        vertex_coordinates[(i * 4) + 2] = attribute.vertices[idx.v_idx * 3 + 2]; // Z
        vertex_coordinates[(i * 4) + 3] = 1.0f;

        /* Get normal data */
        normal_coordinates[(i * 4) + 0] = attribute.normals[idx.vn_idx * 3 + 0]; // X
        normal_coordinates[(i * 4) + 1] = attribute.normals[idx.vn_idx * 3 + 1]; // Y
        normal_coordinates[(i * 4) + 2] = attribute.normals[idx.vn_idx * 3 + 2]; // Z
        normal_coordinates[(i * 4) + 3] = 0.0f;

        /* Get texture data */
        uv_coordinates[(i * 2) + 0] = attribute.texcoords[idx.vt_idx * 2 + 0]; // u
        uv_coordinates[(i * 2) + 1] = attribute.texcoords[idx.vt_idx * 2 + 1]; // v

        for (unsigned int coord = 0; coord < 3; ++coord)
        {
            const float value = vertex_coordinates[(i * 4) + coord];
            bbmin[coord]      = (value < bbmin[coord]) ? value : bbmin[coord];
            bbmax[coord]      = (value > bbmax[coord]) ? value : bbmax[coord];
        }
    }
    printf("bbmin (%f, %f, %f)\n", bbmin[0], bbmin[1], bbmin[2]);
    printf("bbmax (%f, %f, %f)\n", bbmax[0], bbmax[1], bbmax[2]);

    const float height = bbmax[1] - bbmin[1];

    float centerx = -(bbmin[0] + bbmax[0]) * 0.5f;
    float centery = -(bbmin[1] + bbmax[1]) * 0.5f;
    float centerz = -(bbmin[2] + bbmax[2]) * 0.5f;

    ret_obj.transform = mate_translation_make(centerx, centery, centerz);

    float scale_amount = 1.0f / (height);

    printf("height : %f\n", height);
    printf("scale amount : %f\n", scale_amount);

    mmat4 scale = mate_scale_make(scale_amount, scale_amount, scale_amount);

    ret_obj.transform = mate_mat_mul(scale, ret_obj.transform);

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