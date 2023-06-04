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
    argc = 0;
    argv = NULL;

    Reneder_Startup("Rasterizer", SCREEN_WIDTH, SCREEN_HEIGHT);
    atexit(Renderer_Destroy);

    // WOODEN BOX
    // const char *obj_filename = "../../res/Wooden Box/wooden crate.obj";
    // const char *tex_filename = "../../res/Wooden Box/crate_BaseColor.png";
    // const char *nrm_filename = "../../res/Wooden Box/crate_Normal.png";

    // const char *obj_filename = "../../res/Crate/wood_crate.obj";
    // const char *tex_filename = "../../res/Crate/wood_crate_DIFFUSE.png";
    // const char *nrm_filename = "../../res/Crate/wood_crate_NORMAL.png";

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
    const char *obj_filename = "../../res/NormalMappingTestPlane/Normal_Plane.obj";
    const char *tex_filename = "../../res/NormalMappingTestPlane/brickwall.png";
    const char *nrm_filename = "../../res/NormalMappingTestPlane/brickwall_normal.png";
    // const char *tex_filename = "../../res/NormalMappingTestPlane/toy_box_diffuse.png";
    // const char *nrm_filename = "../../res/NormalMappingTestPlane/toy_box_normal.png";

    // CONE
    // const char *obj_filename = "../../res/Cone/Cone.obj";

    // Load texture data
    Texture tex    = Texture_Load(tex_filename, true);
    global_app.tex = tex;

    // Load Normal map
    Texture nrm    = Texture_Load(nrm_filename, true);
    global_app.nrm = nrm;

    Texture_Print_Info(tex);
    Texture_Print_Info(nrm);

    // Load Mesh
    Mesh_Data *mesh;
    Mesh_Get_Data(obj_filename, true, &mesh);

    // Projection Matrix : converts from view space to screen space
    const Mat4x4 Projection_matrix = Get_Projection_Matrix(90.0f, (float)global_renderer.height / (float)global_renderer.width, 0.1f, 1000.0f);

    // Translation Matrix : Move the object in 3D space X Y Z
    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.5f, 2.5f);

    // Camera
    global_app.camera_position = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

    // const float camera_position[3] = {0.0f, 0.0f, 0.0f};
    //  const Mat4x4 View_Matrix        = CreateViewMatrix(camera_position);

    // Lights
    // const Light light = {
    //    //                            A,    B,    G,    R
    //    .ambient_colour  = _mm_set_ps(1.0f, 0.1f, 0.1f, 0.1f),
    //    .diffuse_colour  = _mm_set_ps(1.0f, 0.5f, 0.5f, 0.5f), // first value is index stored at index 3
    //    .specular_colour = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f),
    //    .position        = _mm_set_ps(0.0f, 1.5f, -1.0f, -2.0f),
    //};

    Light light = {
        //                            A,    B,    G,    R     (first value is index stored at index 3)
        .diffuse_colour  = _mm_set_ps(1.0f, 0.0f, 0.5f, 0.0f), // Change this to the texture colour at FragShader stage
        .ambient_amount  = _mm_set1_ps(0.1f),
        .specular_amount = _mm_set1_ps(0.2f),
        .position        = _mm_set_ps(0.0f, -2.0f, -1.0f, 0.5f),
    };

    const float x_adjustment = 0.5f * (float)global_renderer.width;
    const float y_adjustment = 0.5f * (float)global_renderer.height;

    // Loop timer
    struct timer looptimer    = Timer_Init_Start();
    unsigned int loop_counter = 0;
    float        fTheta       = 0.0f; // used for rotation animation

    global_app.shading_mode = NORMAL_MAPPING;

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
        // fTheta = 10.0f;

        const Mat4x4 matRotZ = Get_Rotation_Z_Matrix((float)DEG_to_RAD(180)); // Rotation Z
        const Mat4x4 matRotY = Get_Rotation_Y_Matrix(fTheta);                 // Rotation Y
        // const Mat4x4 matRotY = Get_Rotation_Y_Matrix((float)DEG_to_RAD(85));  // Rotation Y
        const Mat4x4 matRotX = Get_Rotation_X_Matrix(0.0f); // Rotation X

        Matrix_Multiply_Matrix(matRotZ.elements, matRotY.elements, Rotation_Matrix.elements);
        Matrix_Multiply_Matrix(Rotation_Matrix.elements, matRotX.elements, Rotation_Matrix.elements);
        Matrix_Multiply_Matrix(Rotation_Matrix.elements, Translation_matrix.elements, World_Matrix.elements);

        for (size_t i = 0; i < mesh->num_of_triangles; i += 1) // can we jump through triangles?
        {
            __m128 tri1, tri2, tri3;
            tri1 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 0]); // 12 because we load 3 triangles at at a time looping
            tri2 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 4]); // through triangles. 3 traingles each spaced 4 coordinates apart
            tri3 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 8]); // 4 * 3 = 12; [x y z w][x y z w][x y z w]...

            // World_Matrix * Each Vertix = transformed Vertex
            __m128 ws_tri1 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri1);
            __m128 ws_tri2 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri2);
            __m128 ws_tri3 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri3);

            // Calcualte Triangle Area
            // float tri_area = (tri3.m128_f32[0] - tri1.m128_f32[0]) * (tri2.m128_f32[1] - tri1.m128_f32[1]);
            // tri_area       = tri_area - ((tri1.m128_f32[0] - tri2.m128_f32[0]) * (tri1.m128_f32[1] - tri3.m128_f32[1]));
            // tri_area       = 1.0f / tri_area;

            // Vector Dot Product between : Surface normal and CameraRay
            __m128       surface_normal = Calculate_Surface_Normal_SIMD(ws_tri1, ws_tri2, ws_tri3);
            const __m128 camera_ray     = _mm_sub_ps(ws_tri1, global_app.camera_position);

            // Back face culling
            const float dot_product_result = Calculate_Dot_Product_SIMD(surface_normal, camera_ray);
            if (dot_product_result < 0.0f)
            {
                // Calculate the edge values for creating the tangent and bitangent vectors
                // for use with Normal mapping
                __m128 edge1 = _mm_sub_ps(tri2, tri1);
                __m128 edge2 = _mm_sub_ps(tri3, tri1);

                tri1 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, ws_tri1);
                tri2 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, ws_tri2);
                tri3 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, ws_tri3);

                // Camera is at the origin, and we arent moving, so we can skip this?
                // Convert World Space --> View Space
                // tri1 = Matrix_Multiply_Vector_SIMD(View_Matrix.elements, tri1);
                // tri2 = Matrix_Multiply_Vector_SIMD(View_Matrix.elements, tri2);
                // tri3 = Matrix_Multiply_Vector_SIMD(View_Matrix.elements, tri3);

                // Setup texture coordinates

                // Load normal values
                __m128 normal0 = _mm_load_ps(&mesh->normal_coordinates[i * 12 + 0]);
                __m128 normal1 = _mm_load_ps(&mesh->normal_coordinates[i * 12 + 4]);
                __m128 normal2 = _mm_load_ps(&mesh->normal_coordinates[i * 12 + 8]);

                // We need some kind of normal matrix thing
                // Mat4x4 normalMatrix = InverseMat4x4(&World_Matrix);
                // normalMatrix        = TransposeMat4x4(&normalMatrix);
                // normal0 = Matrix_Multiply_Vector_SIMD(normalMatrix.elements, normal0);
                // normal1 = Matrix_Multiply_Vector_SIMD(normalMatrix.elements, normal1);
                // normal2 = Matrix_Multiply_Vector_SIMD(normalMatrix.elements, normal2);

                // Rotation only as we do not change scale or scew
                normal0 = Matrix_Multiply_Vector_SIMD(Rotation_Matrix.elements, normal0);
                normal1 = Matrix_Multiply_Vector_SIMD(Rotation_Matrix.elements, normal1);
                normal2 = Matrix_Multiply_Vector_SIMD(Rotation_Matrix.elements, normal2);

                // tex coordinates are read in like : u u u...
                // uv[0], uv[1], uv[2]
                __m128 texture_v = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 0], mesh->uv_coordinates[6 * i + 2], mesh->uv_coordinates[6 * i + 4]);
                __m128 texture_u = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 1], mesh->uv_coordinates[6 * i + 3], mesh->uv_coordinates[6 * i + 5]);

                Mat3x3 TBN = {0};
                if (global_app.shading_mode == NORMAL_MAPPING)
                {
                    // NORMAL Mapping -----
                    const float deltaUV1_y = mesh->uv_coordinates[6 * i + 3] - mesh->uv_coordinates[6 * i + 1]; // u
                    const float deltaUV1_x = mesh->uv_coordinates[6 * i + 2] - mesh->uv_coordinates[6 * i + 0]; // v

                    // Note, these are flipped
                    const float deltaUV2_y = mesh->uv_coordinates[6 * i + 5] - mesh->uv_coordinates[6 * i + 1]; // u
                    const float deltaUV2_x = mesh->uv_coordinates[6 * i + 4] - mesh->uv_coordinates[6 * i + 0]; // v

                    const float f = 1.0f / (deltaUV1_x * deltaUV2_y - deltaUV2_x * deltaUV1_y);

                    __m128 tangent = _mm_sub_ps(
                        _mm_mul_ps(_mm_set1_ps(deltaUV2_y), edge1),
                        _mm_mul_ps(_mm_set1_ps(deltaUV1_y), edge2));
                    tangent = _mm_mul_ps(_mm_set1_ps(f), tangent);

                    /*
                    the TBN matrix is used to transform vectors from tangent space into world space,
                    while the inverse TBN matrix is used to transform vectors from world space into tangent space.

                    Method 1 - Take the TBN matrix (tangent to world space) give it to the fragment shader,
                                and transform the sampled normal from tangent space (normal map) to world space;
                                the normal is then in the same space as the other lighting variables.
                    Method 2 - Take the inverse of the TBN matrix that transforms any vector from world space to
                                tangent space, and use this matrix to transform not the normal, but the other
                                relevant lighting variables to tangent space; the normal is then again in the
                                same space as the other lighting variables

                    We are using method 2 here... (I hope lol)
                    */
                    // const Mat3x3 m = Mat4x4_to_Mat3x3(World_Matrix);

                    //__m128 N = Mat3x3_mul_m128(m, normal0);
                    //__m128 T = Mat3x3_mul_m128(m, tangent);
                    __m128 T = tangent;
                    __m128 N = normal0;

                    __m128 dotTN        = _mm_dp_ps(T, N, 0x7f);                                                  // dot product of T and N
                    __m128 T_proj_N     = _mm_mul_ps(dotTN, N);                                                   // projection of T onto N
                    __m128 T_perp_N     = _mm_sub_ps(T, T_proj_N);                                                // T component perpendicular to N
                    __m128 T_normalized = _mm_div_ps(T_perp_N, _mm_sqrt_ps(_mm_dp_ps(T_perp_N, T_perp_N, 0x7f))); // normalize T component
                    // T                   = _mm_blend_ps(T_proj_N, T_normalized, 0x80);                             // blend T_proj_N and T_normalized

                    TBN = Create_TBN(T_normalized, N);
                    TBN = TransposeMat3x3(TBN);
                }

                // 3D -> 2D (Projected space)
                // Matrix Projected * Viewed Vertex = projected Vertex
                const float one_over_w1 = 1.0f / tri1.m128_f32[3];
                const float one_over_w2 = 1.0f / tri2.m128_f32[3];
                const float one_over_w3 = 1.0f / tri3.m128_f32[3];

                // Perform x/w, y/w, z/w
                tri1 = _mm_mul_ps(tri1, _mm_set1_ps(one_over_w1));
                tri2 = _mm_mul_ps(tri2, _mm_set1_ps(one_over_w2));
                tri3 = _mm_mul_ps(tri3, _mm_set1_ps(one_over_w3));

                const __m128 texture_w_values = _mm_set_ps(0.0f, one_over_w1, one_over_w2, one_over_w3);
                texture_u                     = _mm_mul_ps(texture_u, texture_w_values);
                texture_v                     = _mm_mul_ps(texture_v, texture_w_values);

                // Sacle Into View (x + 1.0f, y + 1.0f, z + 0.0f, w + 0.0f)
                tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
                tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
                tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));

                tri1 = _mm_mul_ps(tri1, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
                tri2 = _mm_mul_ps(tri2, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
                tri3 = _mm_mul_ps(tri3, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));

                RasterData_t rd = {0};

                rd.light = &light;

                rd.world_space_verticies[0] = ws_tri1;
                rd.world_space_verticies[1] = ws_tri2;
                rd.world_space_verticies[2] = ws_tri3;

                rd.screen_space_verticies[0] = tri1;
                rd.screen_space_verticies[1] = tri2;
                rd.screen_space_verticies[2] = tri3;

                rd.w_values[0] = one_over_w1;
                rd.w_values[1] = one_over_w2;
                rd.w_values[2] = one_over_w3;

                rd.normals[0] = normal0;
                rd.normals[1] = normal1;
                rd.normals[2] = normal2;

                Flat_Shading(rd);
            }
        }

        // Draw_Depth_Buffer(&ren_data);

        // Update Screen
        SDL_UpdateWindowSurface(global_renderer.window);

        if (loop_counter == 32)
        {
            loop_counter               = 0;
            const double frame_time_ms = Timer_Get_Elapsed_MS(&looptimer);
            char         buff[32]      = {0};
            sprintf_s(buff, 32, "%.02f ms/f %s\n", frame_time_ms, Shading_Mode_Str[global_app.shading_mode]);
            SDL_SetWindowTitle(global_renderer.window, buff);
        }
        loop_counter++;
    }

    Mesh_Destroy(mesh);

    return 0;
}