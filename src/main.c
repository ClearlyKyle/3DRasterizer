#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>

#include "SDL2/SDL.h"

#include "Renderer.h"

#define SCREEN_WIDTH  800
#define SCREEN_HEIGHT 800

// TODO: cleanup on exit checking
// TODO: Load 4 triangles, and go through pipeline
// TODO: mt?
// TODO : 4 Triangle loading
// TODO : Organise main better
// TODO : Organise graphics and RenderState better

int main(int argc, char *argv[])
{
    UTILS_UNUSED(argc);
    UTILS_UNUSED(argv);

    Reneder_Startup("Rasterizer", SCREEN_WIDTH, SCREEN_HEIGHT);
    atexit(Renderer_Destroy);

    // WOODEN CRATE
    // const char *obj_filename = "../../res/Wooden Box/wooden crate.obj";

    // TEAPOT
    // const char *obj_filename = "../../res/Teapot/teapot.obj";

    // TEAPOT
    const char *obj_filename = "../../res/Dog House/DogHouse.obj";

    // Load Mesh
    ObjectData_t object = Object_Load(obj_filename);
    global_app.obj      = object;

    global_renderer.max_depth_value = 10.0f;

    // Projection Matrix : converts from view space to screen space
    mmat4 proj = mate_perspective(MATE_D2RF(60.0f), (float)global_renderer.width / (float)global_renderer.height, 0.1f, global_renderer.max_depth_value);

    // Camera
    const vec3 center            = {0.0f, 0.0f, 0.0f};
    const vec3 up                = {0.0f, -1.0f, 0.0f};
    float      camera_z_position = 2.5f;

    global_app.light = (Light_t){
        .ambient_amount  = mate_vec4(0.1f, 0.1f, 0.1f, 1.0f),
        .diffuse_colour  = mate_vec4(0.5f, 0.5f, 0.5f, 1.0f),
        .specular_amount = mate_vec4(0.8f, 0.8f, 0.8f, 1.0f),
        .position        = mate_vec4(1.0f, 0.0f, 6.0f, 0.0f),
    };

    const float x_adjustment    = 0.5f * (float)global_renderer.width;
    const float y_adjustment    = 0.5f * (float)global_renderer.height;
    const mvec4 adjustment      = mate_vec4(x_adjustment, y_adjustment, 1.0f, 1.0f);
    const mvec4 scale_into_view = mate_vec4(1.0f, 1.0f, 0.0f, 0.0f);

    // Loop timer
    struct timer looptimer    = Timer_Init_Start();
    unsigned int loop_counter = 0;
    float        fTheta       = 0.0f; // used for rotation animation

    global_app.shading_mode = SHADING_WIRE_FRAME;

    while (global_renderer.running)
    {
        Renderer_Clear_ZBuffer();
        Renderer_Clear_Screen_Pixels();

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (SDL_QUIT == event.type)
            {
                global_renderer.running = false;
                break;
            }
            else if (SDL_KEYDOWN == event.type && SDL_SCANCODE_D == event.key.keysym.scancode)
            {
                global_app.shading_mode = (global_app.shading_mode + 1) % SHADING_COUNT;

                fprintf_s(stdout, "Shading mode changed to : %s\n", Shading_Mode_Str[global_app.shading_mode]);
            }
            else if (SDL_KEYDOWN == event.type && SDL_SCANCODE_A == event.key.keysym.scancode)
            {
                global_app.shading_mode = (global_app.shading_mode == 0)
                                              ? SHADING_COUNT - 1
                                              : (global_app.shading_mode - 1) % SHADING_COUNT;
                fprintf_s(stdout, "Shading mode changed to : %s\n", Shading_Mode_Str[global_app.shading_mode]);
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
                camera_z_position -= (float)event.wheel.y * 0.1f;
                if (camera_z_position < 1.1f)
                    camera_z_position = 1.1f;
            }
        }

        Timer_Update(&looptimer);

        global_app.camera_position = (mvec4){0.0f, 0.0f, camera_z_position, 0.0f};

        const vec3  eye  = {0.0f, 0.6f, camera_z_position};
        const mmat4 view = mate_look_at(eye, center, up);

        fTheta += (float)Timer_Get_Elapsed_Seconds(&looptimer) / 4;

        const mmat4 rotation_matrix = mate_rotation_make(0.0f, 1.0f, 0.0f, fTheta);
        // const mmat4 rotation_matrix = mate_rotation_make(0.0f, 1.0f, 0.0f, 0.0f);

        // The matrix multiplication is done in the order SRT (Scale, Rotate, and Translate)
        const mmat4 model_matrix      = mate_mat_mul(object.transform, rotation_matrix);
        const mmat4 model_view_matrix = mate_mat_mul(view, model_matrix);
        const mmat4 MVP               = mate_mat_mul(proj, model_view_matrix);

#pragma omp parallel
        {
#pragma omp single
            {
        RasterData_t       rd[4]                         = {0};
        uint8_t            number_of_collected_triangles = 0;
        const unsigned int number_of_triangles           = object.number_of_triangles;

        for (size_t i = 0; i <= object.number_of_triangles; ++i) // can we jump through triangles?
        {
            mvec4 raw_v0 = mate_vec4_load(&object.vertex_coordinates[i * 12 + 0]); // 12 because we load 3 triangles at at a time looping
            mvec4 raw_v1 = mate_vec4_load(&object.vertex_coordinates[i * 12 + 4]); // through triangles. 3 traingles each spaced 4 coordinates apart
            mvec4 raw_v2 = mate_vec4_load(&object.vertex_coordinates[i * 12 + 8]); // 4 * 3 = 12; [x y z w][x y z w][x y z w]...

            mvec4 ws_tri0 = mate_mat_mulv(model_matrix, raw_v0);
            mvec4 ws_tri1 = mate_mat_mulv(model_matrix, raw_v1);
            mvec4 ws_tri2 = mate_mat_mulv(model_matrix, raw_v2);

            mvec4 proj_v0 = mate_mat_mulv(MVP, raw_v0);
            mvec4 proj_v1 = mate_mat_mulv(MVP, raw_v1);
            mvec4 proj_v2 = mate_mat_mulv(MVP, raw_v2);

            // 3D -> 2D (Projected space)
            // Matrix Projected * Viewed Vertex = projected Vertex
            const float w0 = proj_v0.f[3];
            const float w1 = proj_v1.f[3];
            const float w2 = proj_v2.f[3];

            // Clipping (not the best, but its something)
            if (!(-w0 <= proj_v0.f[0] && proj_v0.f[0] <= w0) || !(-w0 <= proj_v0.f[1] && proj_v0.f[1] <= w0) || !(-w0 <= proj_v0.f[2] && proj_v0.f[2] <= w0))
                continue;
            if (!(-w1 <= proj_v1.f[0] && proj_v1.f[0] <= w1) || !(-w1 <= proj_v1.f[1] && proj_v1.f[1] <= w1) || !(-w1 <= proj_v1.f[2] && proj_v1.f[2] <= w1))
                continue;
            if (!(-w2 <= proj_v2.f[0] && proj_v2.f[0] <= w2) || !(-w2 <= proj_v2.f[1] && proj_v2.f[1] <= w2) || !(-w2 <= proj_v2.f[2] && proj_v2.f[2] <= w2))
                continue;

            const float one_over_w0 = 1.0f / w0;
            const float one_over_w1 = 1.0f / w1;
            const float one_over_w2 = 1.0f / w2;

            // Perform x/w, y/w, z/w
            proj_v0 = mate_vec4_scale(proj_v0, one_over_w0);
            proj_v1 = mate_vec4_scale(proj_v1, one_over_w1);
            proj_v2 = mate_vec4_scale(proj_v2, one_over_w2);

            // Sacle Into View (x + 1.0f, y + 1.0f, z + 1.0f, w + 0.0f)
            proj_v0 = mate_vec4_add(proj_v0, scale_into_view);
            proj_v1 = mate_vec4_add(proj_v1, scale_into_view);
            proj_v2 = mate_vec4_add(proj_v2, scale_into_view);

            proj_v0 = mate_vec4_mul(proj_v0, adjustment);
            proj_v1 = mate_vec4_mul(proj_v1, adjustment);
            proj_v2 = mate_vec4_mul(proj_v2, adjustment);

            // Calcualte Triangle Area
            // const float B1 = proj_v2.f[0] - proj_v0.f[0];
            // const float B2 = proj_v0.f[0] - proj_v1.f[0];
            // const float A1 = proj_v0.f[1] - proj_v2.f[1];
            // const float A2 = proj_v1.f[1] - proj_v0.f[1];

            // mvec4 AB = mate_vec4_sub(proj_v0, proj_v1);
            // mvec4 AC = mate_vec4_sub(proj_v0, proj_v2);

            // float sign = AB.f[0] * AC.f[1] - AC.f[0] * AB.f[1];

            // Back face culling
            // if (sign > 0.0f)
            {
                // Calculate the edge values for creating the tangent and bitangent vectors
                // for use with Normal mapping
                __m128 edge1 = _mm_sub_ps(raw_v1.m, raw_v0.m);
                __m128 edge2 = _mm_sub_ps(raw_v2.m, raw_v0.m);

                // Load normal values
                mvec4 raw_nrm2 = mate_vec4_load(&object.normal_coordinates[i * 12 + 0]);
                mvec4 raw_nrm1 = mate_vec4_load(&object.normal_coordinates[i * 12 + 4]);
                mvec4 raw_nrm0 = mate_vec4_load(&object.normal_coordinates[i * 12 + 8]);

                /* Construct the Normal Matrix*/
                mmat3 nrm_matrix = mate_mat3_inv(mate_mat4_make_mat3_transpose(model_matrix));

                vec3 new_n0, new_n1, new_n2;
                mate_mat3_mulv(nrm_matrix, (vec3){raw_nrm0.f[0], raw_nrm0.f[1], raw_nrm0.f[2]}, new_n2);
                mate_mat3_mulv(nrm_matrix, (vec3){raw_nrm1.f[0], raw_nrm1.f[1], raw_nrm1.f[2]}, new_n1);
                mate_mat3_mulv(nrm_matrix, (vec3){raw_nrm2.f[0], raw_nrm2.f[1], raw_nrm2.f[2]}, new_n0);

                mvec4 ws_nrm0 = {.f = {new_n0[0], new_n0[1], new_n0[2], 0.0f}};
                mvec4 ws_nrm1 = {.f = {new_n1[0], new_n1[1], new_n1[2], 0.0f}};
                mvec4 ws_nrm2 = {.f = {new_n2[0], new_n2[1], new_n2[2], 0.0f}};

                // tex coordinates are read in like : u u u...
                // uv[0], uv[1], uv[2]
                if (object.has_texcoords)
                {
                    __m128 texture_u = _mm_setr_ps(object.uv_coordinates[6 * i + 0], object.uv_coordinates[6 * i + 2], object.uv_coordinates[6 * i + 4], 0.0f);
                    __m128 texture_v = _mm_setr_ps(object.uv_coordinates[6 * i + 1], object.uv_coordinates[6 * i + 3], object.uv_coordinates[6 * i + 5], 0.0f);

                    rd[number_of_collected_triangles].tex_u = (mvec4){.m = texture_u};
                    rd[number_of_collected_triangles].tex_v = (mvec4){.m = texture_v};
                }

                // mmat3 TBN = {0};
                if (global_app.shading_mode == SHADING_NORMAL_MAPPING)
                {
                    continue; // TMP

                    // NORMAL Mapping -----
                    const float deltaUV1_y = object.uv_coordinates[6 * i + 3] - object.uv_coordinates[6 * i + 1]; // u
                    const float deltaUV1_x = object.uv_coordinates[6 * i + 2] - object.uv_coordinates[6 * i + 0]; // v

                    // Note, these are flipped
                    const float deltaUV2_y = object.uv_coordinates[6 * i + 5] - object.uv_coordinates[6 * i + 1]; // u
                    const float deltaUV2_x = object.uv_coordinates[6 * i + 4] - object.uv_coordinates[6 * i + 0]; // v

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
                    __m128 N = raw_nrm0.m;

                    __m128 dotTN        = _mm_dp_ps(T, N, 0x7f);                                                  // dot product of T and N
                    __m128 T_proj_N     = _mm_mul_ps(dotTN, N);                                                   // projection of T onto N
                    __m128 T_perp_N     = _mm_sub_ps(T, T_proj_N);                                                // T component perpendicular to N
                    __m128 T_normalized = _mm_div_ps(T_perp_N, _mm_sqrt_ps(_mm_dp_ps(T_perp_N, T_perp_N, 0x7f))); // normalize T component
                                                                                                                  // T                   = _mm_blend_ps(T_proj_N, T_normalized, 0x80);                             // blend T_proj_N and T_normalized

                    UTILS_UNUSED(T_normalized); // TODO : Move this to a "core"

                    // TBN = Create_TBN(T_normalized, N);
                    // TBN = TransposeMat3x3(TBN);
                }

                mvec4 end0 = mate_vec4_add3(raw_v0, mate_vec4_scale(raw_nrm0, 0.4f));
                mvec4 end1 = mate_vec4_add3(raw_v1, mate_vec4_scale(raw_nrm1, 0.4f));
                mvec4 end2 = mate_vec4_add3(raw_v2, mate_vec4_scale(raw_nrm2, 0.4f));

                end0 = mate_mat_mulv(MVP, end0);
                end1 = mate_mat_mulv(MVP, end1);
                end2 = mate_mat_mulv(MVP, end2);

                const float end_w1 = 1.0f / mate_vec4_get(end0, 3);
                const float end_w2 = 1.0f / mate_vec4_get(end1, 3);
                const float end_w3 = 1.0f / mate_vec4_get(end2, 3);

                // Perform x/w, y/w, z/w
                end0 = mate_vec4_scale(end0, end_w1);
                end1 = mate_vec4_scale(end1, end_w2);
                end2 = mate_vec4_scale(end2, end_w3);

                end0 = mate_vec4_add(end0, scale_into_view);
                end1 = mate_vec4_add(end1, scale_into_view);
                end2 = mate_vec4_add(end2, scale_into_view);

                end0 = mate_vec4_mul(end0, adjustment);
                end1 = mate_vec4_mul(end1, adjustment);
                end2 = mate_vec4_mul(end2, adjustment);

                rd[number_of_collected_triangles].world_space_verticies[0] = ws_tri0;
                rd[number_of_collected_triangles].world_space_verticies[1] = ws_tri1;
                rd[number_of_collected_triangles].world_space_verticies[2] = ws_tri2;

                rd[number_of_collected_triangles].screen_space_verticies[0] = proj_v0;
                rd[number_of_collected_triangles].screen_space_verticies[1] = proj_v1;
                rd[number_of_collected_triangles].screen_space_verticies[2] = proj_v2;

                rd[number_of_collected_triangles].endpoints[0] = end0;
                rd[number_of_collected_triangles].endpoints[1] = end1;
                rd[number_of_collected_triangles].endpoints[2] = end2;

                rd[number_of_collected_triangles].w_values[0] = one_over_w0; // NOTE : Do we need to store these?
                rd[number_of_collected_triangles].w_values[1] = one_over_w1;
                rd[number_of_collected_triangles].w_values[2] = one_over_w2;

                rd[number_of_collected_triangles].normals[0] = ws_nrm0;
                rd[number_of_collected_triangles].normals[1] = ws_nrm1;
                rd[number_of_collected_triangles].normals[2] = ws_nrm2;

                number_of_collected_triangles++;

                if (number_of_collected_triangles < 4 && i < number_of_triangles)
                    continue;

#pragma omp task
                        {
                            // Time the task
                            // const double start_time = omp_get_wtime();

                Flat_Shading(rd, number_of_collected_triangles);

                            // const double end_time     = omp_get_wtime();
                            // const double elapsed_time = end_time - start_time;
                            // printf("Thread (%d) - task execution time: %f seconds\n", omp_get_thread_num(), elapsed_time);
                        }
                number_of_collected_triangles = 0;
            }
        }
#pragma omp taskwait
            }
        }
        if (global_app.shading_mode == SHADING_DEPTH_BUFFER)
            Draw_Depth_Buffer();

        // Update Screen
        SDL_UpdateWindowSurface(global_renderer.window);

        if (loop_counter >= 64)
        {
            char buff[64] = {0};
            sprintf_s(buff, 64, "%.02f ms/f %s\n", loop_accumulated_timer / loop_counter, Shading_Mode_Str[global_app.shading_mode]);
            SDL_SetWindowTitle(global_renderer.window, buff);

            loop_counter           = 0;
            loop_accumulated_timer = 0.0;
        }
        loop_accumulated_timer += Timer_Get_Elapsed_MS(&looptimer);
        loop_counter++;
    }

    Object_Destroy(&object);

    return 0;
}