#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "SDL2/SDL.h"

#include "Renderer.h"
#include "ObjLoader.h"

#define SCREEN_WIDTH  1000
#define SCREEN_HEIGHT 900

// TODO: detect what files are being loaded
// TODO: cleanup on exit checking
// TODO: set render mode
// TODO: Improve object loading and setting object data
// TODO: Better matrix multiplication

int main(int argc, char *argv[])
{
    Reneder_Startup("Rasterizer", SCREEN_WIDTH, SCREEN_HEIGHT);
    atexit(Renderer_Destroy);

    // WOODEN BOX
    const char *obj_filename = "../../res/Wooden Box/wooden crate.obj";
    // const char *obj_filename = "../../res/Wooden Box/box_triange.obj";
    const char *tex_filename = "../../res/Wooden Box/crate_BaseColor.png";
    const char *nrm_filename = "../../res/Wooden Box/crate_Normal.png";

    // TEAPOT
    // const char *obj_filename = "../../res/Teapot/teapot2.obj";

    // DOG HOUSE
    // const char *obj_filename = "../../res/Dog House/Doghouse.obj";
    // const char *tex_filename = "../../res/Dog House/Doghouse_PBR_BaseColor.png";
    // const char *nrm_filename = "../../res/Dog House/Doghouse_PBR_Normal.png";

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

    // Load texture data
    Texture tex    = Texture_Load(tex_filename);
    global_app.tex = tex;

    // Load Normal map
    Texture nrm    = Texture_Load(nrm_filename);
    global_app.nrm = nrm;

    // Load Mesh
    Mesh_Data *mesh;
    Mesh_Get_Data(obj_filename, true, &mesh);

    // Projection Matrix : converts from view space to screen space
    const Mat4x4 Projection_matrix = Get_Projection_Matrix(90.0f, (float)global_renderer.height / (float)global_renderer.width, 0.1f, 1000.0f);

    // Translation Matrix : Move the object in 3D space X Y Z
    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.5f, 3.5f);

    // Camera
    global_app.camera_position = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

    // Lights
    const Light light = {
        //                            A,    B,    G,    R
        .ambient_colour  = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f),
        .diffuse_colour  = _mm_set_ps(1.0f, 0.0f, 1.0f, 0.0f), // first value is index stored at index 3
        .specular_colour = _mm_set_ps(1.0f, 1.0f, 0.0f, 0.0f),
        .position        = _mm_set_ps(0.0f, 1.0f, 0.0f, -1.0f),
    };

    const float x_adjustment = 0.5f * (float)global_renderer.width;
    const float y_adjustment = 0.5f * (float)global_renderer.height;

    // Loop timer
    struct timer looptimer    = Timer_Init_Start();
    unsigned int loop_counter = 0;
    float        fTheta       = 0.0f; // used for rotation animation

    global_app.shading_mode = TEXTURED;

    while (global_renderer.running)
    {
        // Clear Z Buffer
        Renderer_Clear_ZBuffer();

        // Clear Pixels
        Renderer_Clear_Screen_Pixels();

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (SDL_QUIT == event.type)
            {
                global_renderer.running = false;
                break;
            }
        }

        Timer_Update(&looptimer);

        Mat4x4 World_Matrix    = {0.0f};
        Mat4x4 Rotation_Matrix = {0.0f};

        fTheta += (float)Timer_Get_Elapsed_Seconds(&looptimer) / 4;
        // fTheta = 0.0f;

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
            __m128       surface_normal = Calculate_Surface_Normal_SIMD(tri1, tri2, tri3);
            const __m128 camera_ray     = _mm_sub_ps(tri1, global_app.camera_position);

            // Back face culling
            const float dot_product_result = Calculate_Dot_Product_SIMD(surface_normal, camera_ray);
            if (dot_product_result < 0.0f)
            {
                surface_normal                           = Normalize_m128(surface_normal);
                const __m128 world_position_verticies[3] = {tri1, tri2, tri3};

                __m128 edge1      = _mm_sub_ps(tri2, tri1);
                __m128 edge2      = _mm_sub_ps(tri3, tri1);
                edge1.m128_f32[3] = 0.0f;
                edge2.m128_f32[3] = 0.0f;

                // 3D -> 2D (Projected space)
                // Matrix Projected * Viewed Vertex = projected Vertex
                tri1 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri1);
                tri2 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri2);
                tri3 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri3);

                // Setup texture coordinates
                const float one_over_w1 = 1.0f / tri1.m128_f32[3];
                const float one_over_w2 = 1.0f / tri2.m128_f32[3];
                const float one_over_w3 = 1.0f / tri3.m128_f32[3];

                const float w_values[3] = {one_over_w1, one_over_w2, one_over_w3};

                // tex coordinates are read in like : u u u...
                // uv[0], uv[1], uv[2]
                __m128 texture_v = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 0], mesh->uv_coordinates[6 * i + 2], mesh->uv_coordinates[6 * i + 4]);
                __m128 texture_u = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 1], mesh->uv_coordinates[6 * i + 3], mesh->uv_coordinates[6 * i + 5]);

                // NORMAL Mapping
                const float deltaUV1_x = mesh->uv_coordinates[6 * i + 3] - mesh->uv_coordinates[6 * i + 1];
                const float deltaUV1_y = mesh->uv_coordinates[6 * i + 2] - mesh->uv_coordinates[6 * i + 0];

                const float deltaUV2_x = mesh->uv_coordinates[6 * i + 5] - mesh->uv_coordinates[6 * i + 1];
                const float deltaUV2_y = mesh->uv_coordinates[6 * i + 4] - mesh->uv_coordinates[6 * i + 0];

                const float f = 1.0f / (deltaUV1_x * deltaUV2_y - deltaUV2_x * deltaUV1_y);

                __m128 tangent = _mm_sub_ps(
                    _mm_mul_ps(_mm_set1_ps(deltaUV2_y), edge1),
                    _mm_mul_ps(_mm_set1_ps(deltaUV1_y), edge2));
                tangent = _mm_mul_ps(_mm_set1_ps(f), tangent);

                const __m128 texture_w_values = _mm_set_ps(0.0f, w_values[0], w_values[1], w_values[2]);
                texture_u                     = _mm_mul_ps(texture_u, texture_w_values);
                texture_v                     = _mm_mul_ps(texture_v, texture_w_values);

                // Perform x/w, y/w, z/w
                tri1 = _mm_mul_ps(tri1, _mm_set1_ps(one_over_w1));
                tri2 = _mm_mul_ps(tri2, _mm_set1_ps(one_over_w2));
                tri3 = _mm_mul_ps(tri3, _mm_set1_ps(one_over_w3));

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
                const __m128 normal_coordinates[3]        = {normal0, normal1, normal2};

                if (global_app.shading_mode == WIRE_FRAME)
                    Draw_Triangle_Outline(screen_position_verticies, (SDL_Color){255, 255, 255, 255});
                else if (global_app.shading_mode < TEXTURED)
                    Flat_Shading(screen_position_verticies, world_position_verticies, w_values, normal_coordinates, &light);
                else
                    Textured_Shading(screen_position_verticies, world_position_verticies, w_values, normal_coordinates, texture_u, texture_v, surface_normal, TBN, &light);
            }
        }

        // Draw_Depth_Buffer(&ren_data);

        // Update Screen
        SDL_UpdateWindowSurface(global_renderer.window);

        if (loop_counter == 32)
        {
            loop_counter               = 0;
            const double frame_time_ms = Timer_Get_Elapsed_MS(&looptimer);
            char         buff[16]      = {0};
            sprintf_s(buff, 16, "%.02f ms/f\n", frame_time_ms);
            SDL_SetWindowTitle(global_renderer.window, buff);
        }
        loop_counter++;
    }

    Mesh_Destroy(mesh);

    return 0;
}