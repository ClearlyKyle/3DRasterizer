#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#include "SDL2/SDL.h"

#include "lights.h"
#include "Renderer.h"
#include "vector.h"
#include "test_sqaure.h"
#include "ObjLoader.h"

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 900

int main(int argc, char *argv[])
{
    Renderer ren = SDL_Startup("Window", SCREEN_WIDTH, SCREEN_HEIGHT);

    SDL_Surface *window_surface = SDL_GetWindowSurface(ren.window);

    // WOODEN BOX
    const char *obj_filename = "../../res/Wooden Box/wooden crate.obj";
    //// const char *obj_filename = "../../res/Wooden Box/box_triange.obj";
    // const char *tex_filename = "../../res/Wooden Box/crate_BaseColor.png";
    // const char *nrm_filename = "../../res/Wooden Box/crate_Normal.png";

    // DOG HOUSE
    // const char *obj_filename = "../../res/Dog House/Doghouse.obj";
    const char *tex_filename = "../../res/Dog House/Doghouse_PBR_BaseColor.png";
    const char *nrm_filename = "../../res/Dog House/Doghouse_PBR_Normal.png";

    // CRATE
    // const char *obj_filename = "../../res/Crate/cube_triang.obj";
    // const char *obj_filename = "../../res/Crate/Crate1.obj";
    // const char *obj_filename = "../../res/Crate/Crate_N_T.obj";
    // const char *obj_filename = "../../res/Crate/cube_normals.obj";
    // const char *tex_filename = "../../res/Crate/crate_1.png";

    // SPHERE
    // const char *obj_filename = "../../res/Sphere/simple_sphereobj.obj";
    // const char *obj_filename = "../../res/Sphere/low_poly_sphere.obj";
    // const char *obj_filename = "../../res/Sphere/sphere_normals.obj";
    // const char *obj_filename = "../../res/Sphere/sphere_smooth.obj";

    // CYLINDER
    // const char *obj_filename = "../../res/Sphere/simple_sphereobj.obj";
    // const char *obj_filename = "../../res/Sphere/low_poly_sphere.obj";
    // const char *obj_filename = "../../res/Cylinder/cylinder_normals.obj";

    // PLANE
    // const char *obj_filename = "../../res/NormalMappingTestPlane/Normal_Plane.obj";
    // const char *tex_filename = "../../res/NormalMappingTestPlane/brickwall.png";
    // const char *nrm_filename = "../../res/NormalMappingTestPlane/brickwall_normal.png";

    // CONE
    // const char *obj_filename = "../../res/Cone/Cone.obj";

    // textures oriented tha same as you view them in paint
    stbi_set_flip_vertically_on_load(1);

    // Load texture data
    int tex_w, tex_h, tex_bpp;
    // unsigned char *texture_data = stbi_load(tex_filename, &tex_w, &tex_h, &tex_bpp, STBI_rgb);
    unsigned char *texture_data = stbi_load(tex_filename, &tex_w, &tex_h, &tex_bpp, STBI_rgb);
    if (!texture_data)
    {
        fprintf(stderr, "ERROR! Loading image : %s\n", stbi_failure_reason());
        return 0;
    }

    fprintf(stderr, "Texture Loaded : %s\n", tex_filename);
    fprintf(stderr, "Texture width  : %d\n", tex_w);
    fprintf(stderr, "Texture height : %d\n", tex_h);
    fprintf(stderr, "Texture bbp    : %d\n", tex_bpp);

    // Load Normal map
    int nrm_w, nrm_h, nrm_bpp;
    unsigned char *normal_data = stbi_load(nrm_filename, &nrm_w, &nrm_h, &nrm_bpp, STBI_rgb);
    if (!texture_data)
    {
        fprintf(stderr, "Loading image : %s\n", stbi_failure_reason());
        return 0;
    }

    fprintf(stderr, "Normal Map Loaded : %s\n", nrm_filename);
    fprintf(stderr, "Normal Map width  : %d\n", nrm_w);
    fprintf(stderr, "Normal Map height : %d\n", nrm_h);
    fprintf(stderr, "Normal Map bbp    : %d\n", nrm_bpp);

    Rendering_data ren_data;

    // Get window data
    ren_data.surface = window_surface;
    ren_data.fmt = window_surface->format;
    ren_data.pixels = window_surface->pixels;
    ren_data.screen_height = window_surface->h;
    ren_data.screen_width = window_surface->w;
    ren_data.screen_num_pixels = window_surface->h * window_surface->w;

    // Texture Data
    ren_data.tex_data = texture_data;
    ren_data.tex_w = tex_w;
    ren_data.tex_h = tex_h;
    ren_data.tex_bpp = tex_bpp;

    ren_data.nrm_data = normal_data;
    ren_data.nrm_w = nrm_w;
    ren_data.nrm_h = nrm_h;
    ren_data.nrm_bpp = nrm_bpp;

    // Normal Map

    // Allocate z buffer
    float *z_buff = (float *)_aligned_malloc(sizeof(float) * ren_data.screen_num_pixels, 16);
    ren_data.z_buffer_array = z_buff;
    ren_data.max_depth_value = 10.0f;

    // Lights (W, Z, Y, X)
    ren_data.light_position = _mm_set_ps(0.0f, 1.0f, 4.0f, 1.0f);
    ren_data.light_value = 0.0f;

    // Load Mesh
    Mesh_Data *mesh;
    Get_Object_Data(obj_filename, true, &mesh);

    // Projection Matrix : converts from view space to screen space
    const Mat4x4 Projection_matrix = Get_Projection_Matrix(90.0f, (float)ren.HEIGHT / (float)ren.WIDTH, 0.1f, 1000.0f);

    // Translation Matrix : Move the object in 3D space X Y Z
    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.5f, 3.5f);

    // Camera
    const __m128 camera_position = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    // const __m128 camera_direction = _mm_set_ps(0.0f, -1.0f, 0.0f, 0.0f);

    // Lights -------------------------
    Fragment frag;
    const float ambient_strength = 0.1f;
    const float shininess = 64.0f;
    const __m128 light_colour = _mm_set1_ps(1.0f);
    frag.ambient = _mm_mul_ps(light_colour, _mm_set1_ps(ambient_strength));

    // Point Light
    // const PointLight point_light = Get_Point_Light(-2.0f, -2.0f, -10.0f, 1.0f, 0.045f, 0.0075f);
    const PointLight point_light = Get_Point_Light(-1.0f, 0.0f, 1.0f, 1.0f, 0.045f, 0.0075f);

    const __m128 object_colour = _mm_set_ps(255.0f, 000.0f, 255.0f, 255.0f);
    frag.color = _mm_mul_ps(object_colour, frag.ambient);

    // Line colour for wire model drawing
    const SDL_Colour LINE_COLOUR = {255, 255, 255, 255};

    const float x_adjustment = 0.5f * (float)ren.WIDTH;
    const float y_adjustment = 0.5f * (float)ren.HEIGHT;

    // Performance counters
    Uint64 LastCounter = SDL_GetPerformanceCounter();
    double MSPerFrame = 0.0;
    float fTheta = 0.0f;
    char window_title[32];

    ren.running = true;
    unsigned int loop_counter = 0;
    unsigned int shading_switch_counter = 1;

    ren_data.shading = WIRE_FRAME;

    while (ren.running)
    {
        // Clear Z Buffer
        // memset(ren_data.z_buffer_array, 0xF, ren_data.screen_num_pixels * 4);
        __m128 MAX_DEPTH = _mm_set1_ps(ren_data.max_depth_value);
        for (__m128 *i = (__m128 *)ren_data.z_buffer_array, *end = (__m128 *)&ren_data.z_buffer_array[ren_data.screen_num_pixels];
             i < end;
             i += 1)
        {
            *i = MAX_DEPTH;
        }

        // Clear Pixels
        memset(ren_data.pixels, 0, ren_data.screen_num_pixels * 4);

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (SDL_QUIT == event.type)
            {
                ren.running = false;
                break;
            }
        }

        // Automated Switching Shading mode
        if (shading_switch_counter % 2000 == 0)
        {
            shading_switch_counter = 0;
            ren_data.shading++;

            if (ren_data.shading == SHADING_COUNT)
            {
                ren_data.shading = WIRE_FRAME;
            }
        }
        shading_switch_counter += 1;

        Mat4x4 World_Matrix = {0.0f};
        Mat4x4 Rotation_Matrix = {0.0f};

        fTheta += (float)MSPerFrame / 3;
        // fTheta += 0.0f;

        const Mat4x4 matRotZ = Get_Rotation_Z_Matrix((float)DEG_to_RAD(180)); // Rotation Z
        const Mat4x4 matRotY = Get_Rotation_Y_Matrix(fTheta);                 // Rotation Y
        const Mat4x4 matRotX = Get_Rotation_X_Matrix(0.0f);                   // Rotation X

        Matrix_Multiply_Matrix(matRotZ.elements, matRotY.elements, Rotation_Matrix.elements);
        Matrix_Multiply_Matrix(Rotation_Matrix.elements, matRotX.elements, Rotation_Matrix.elements);
        Matrix_Multiply_Matrix(Rotation_Matrix.elements, Translation_matrix.elements, World_Matrix.elements);

        for (size_t i = 0; i < mesh->num_of_triangles; i += 1) // can we jump through triangles?
        {
            __m128 tri1 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 0]); // 12 because we load 3 triangles at at a time looping
            __m128 tri2 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 4]); // through triangles. 3 traingles each spaced 4 coordinates apart
            __m128 tri3 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 8]); // 4 * 3 = 12; [x y z w][x y z w][x y z w]...

            // World_Matrix * Each Vertix = transformed Vertex
            tri1 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri1);
            tri2 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri2);
            tri3 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri3);

            // Vector Dot Product between : Surface normal and CameraRay
            __m128 surface_normal = Calculate_Surface_Normal_SIMD(tri1, tri2, tri3);
            const __m128 camera_ray = _mm_sub_ps(tri1, camera_position);

            // Back face culling
            // TODO : Change this to work with normals
            const float dot_product_result = Calculate_Dot_Product_SIMD(surface_normal, camera_ray);
            if (dot_product_result < 0.0f)
            {
                surface_normal = Normalize_m128(surface_normal);
                const __m128 world_position_verticies[3] = {tri1, tri2, tri3};

                __m128 edge1 = _mm_sub_ps(tri2, tri1);
                __m128 edge2 = _mm_sub_ps(tri3, tri1);

                // 3D -> 2D
                // Matrix Projected * Viewed Vertex = projected Vertex
                tri1 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri1);
                tri2 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri2);
                tri3 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri3);

                // Setup texture coordinates
                const __m128 one_over_w1 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(tri1, tri1, _MM_SHUFFLE(3, 3, 3, 3)));
                const __m128 one_over_w2 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(tri2, tri2, _MM_SHUFFLE(3, 3, 3, 3)));
                const __m128 one_over_w3 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(tri3, tri3, _MM_SHUFFLE(3, 3, 3, 3)));

                const __m128 w_values[3] = {one_over_w1, one_over_w2, one_over_w3};

                // tex coordinates are read in like : u u u...
                // uv[0], uv[1], uv[2]
                __m128 texture_v = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 0], mesh->uv_coordinates[6 * i + 2], mesh->uv_coordinates[6 * i + 4]);
                __m128 texture_u = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 1], mesh->uv_coordinates[6 * i + 3], mesh->uv_coordinates[6 * i + 5]);

                // NORMAL Mapping
                edge1.m128_f32[3] = 0.0f;
                edge2.m128_f32[3] = 0.0f;

                const float deltaUV1_x = mesh->uv_coordinates[6 * i + 3] - mesh->uv_coordinates[6 * i + 1];
                const float deltaUV1_y = mesh->uv_coordinates[6 * i + 2] - mesh->uv_coordinates[6 * i + 0];

                const float deltaUV2_x = mesh->uv_coordinates[6 * i + 5] - mesh->uv_coordinates[6 * i + 1];
                const float deltaUV2_y = mesh->uv_coordinates[6 * i + 4] - mesh->uv_coordinates[6 * i + 0];

                const float f = 1.0f / (deltaUV1_x * deltaUV2_y - deltaUV2_x * deltaUV1_y);

                __m128 tangent = _mm_sub_ps(
                    _mm_mul_ps(_mm_set1_ps(deltaUV2_y), edge1),
                    _mm_mul_ps(_mm_set1_ps(deltaUV1_y), edge2));
                tangent = _mm_mul_ps(_mm_set1_ps(f), tangent);

                const __m128 texture_w_values = _mm_set_ps(0.0f, one_over_w1.m128_f32[0], one_over_w2.m128_f32[0], one_over_w3.m128_f32[0]);
                texture_u = _mm_mul_ps(texture_u, texture_w_values);
                texture_v = _mm_mul_ps(texture_v, texture_w_values);

                // Perform x/w, y/w, z/w
                tri1 = _mm_mul_ps(tri1, one_over_w1);
                tri2 = _mm_mul_ps(tri2, one_over_w2);
                tri3 = _mm_mul_ps(tri3, one_over_w3);

                // Sacle Into View
                tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
                tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
                tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));

                tri1 = _mm_mul_ps(tri1, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
                tri2 = _mm_mul_ps(tri2, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
                tri3 = _mm_mul_ps(tri3, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));

                // similar procedure for calculating tangent/bitangent for plane's second triangle

                __m128 normal0 = _mm_load_ps(&mesh->normal_coordinates[i * 12 + 0]);
                __m128 normal1 = _mm_load_ps(&mesh->normal_coordinates[i * 12 + 4]);
                __m128 normal2 = _mm_load_ps(&mesh->normal_coordinates[i * 12 + 8]);

                const Mat4x4 TBN = Get_TBN_Matrix(tangent, normal0, World_Matrix);

                // Rotation only as we do not change scale or scew
                normal0 = Matrix_Multiply_Vector_SIMD(Rotation_Matrix.elements, normal0);
                normal1 = Matrix_Multiply_Vector_SIMD(Rotation_Matrix.elements, normal1);
                normal2 = Matrix_Multiply_Vector_SIMD(Rotation_Matrix.elements, normal2);

                // Screen Vertex Postions
                const __m128 screen_position_verticies[3] = {tri1, tri2, tri3};
                const __m128 normal_coordinates[3] = {normal0, normal1, normal2};

                if (ren_data.shading == WIRE_FRAME)
                {
                    Draw_Triangle_Outline(&ren_data, screen_position_verticies, &LINE_COLOUR);
                }
                else if (ren_data.shading >= BLIN_PHONG)
                {
                    Flat_Shading(&ren_data, screen_position_verticies, world_position_verticies, w_values, normal_coordinates, surface_normal, &point_light, ren_data.shading);
                }
                else
                {
                    Textured_Shading(&ren_data, screen_position_verticies, world_position_verticies, w_values, normal_coordinates, texture_u, texture_v, surface_normal, &point_light, TBN);
                }
            }
        }

        // Draw_Depth_Buffer(&ren_data);

        // Update Screen
        SDL_UpdateWindowSurface(ren.window);

        // End frame timing
        const Uint64 EndCounter = SDL_GetPerformanceCounter();
        const Uint64 CounterElapsed = EndCounter - LastCounter;

        MSPerFrame = ((double)CounterElapsed / (double)SDL_GetPerformanceFrequency());

        if (loop_counter == 32)
        {
            loop_counter = 0;
            const double FPS = (double)CounterElapsed / (double)SDL_GetPerformanceFrequency();
            snprintf(window_title, sizeof(window_title), "%.02f ms/f \t%.02f f/s\n", 1000.0 * MSPerFrame, FPS);
            SDL_SetWindowTitle(ren.window, window_title);
        }
        else
        {
            loop_counter++;
        }

        LastCounter = EndCounter;
    }

    Free_Mesh(&mesh);
    // free(ren_data.pixels);
    // free(ren_data.z_buffer_array);
    SDL_CleanUp(&ren);
    return 0;
}