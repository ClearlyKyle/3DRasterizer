#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#include "SDL2/SDL.h"

#include "vector.h"
#include "Renderer.h"
#include "test_sqaure.h"
#include "ObjLoader.h"

#define SCREEN_WIDTH 900
#define SCREEN_HEIGHT 1000

int main(int argc, char *argv[])
{
    Renderer ren = SDL_Startup("Window", SCREEN_HEIGHT, SCREEN_WIDTH);

    SDL_Surface *window_surface = SDL_GetWindowSurface(ren.window);

    // const char *obj_filename = "../../res/Wooden Box/wooden crate.obj";
    //  const char *obj_filename = "../../res/Wooden Box/wooden crate triangulated.obj";

    // CRATE
    // const char *obj_filename = "../../res/Crate/cube_triang.obj";
    const char *obj_filename = "../../res/Crate/Crate1.obj";
    const char *tex_filename = "../../res/Crate/crate_1.png";

    // Load texture data
    int tex_w, tex_h, tex_bpp;
    unsigned char *texture_data = stbi_load(tex_filename, &tex_w, &tex_h, &tex_bpp, STBI_rgb_alpha);
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
    // int nrm_w, nrm_h, nrm_bpp;
    // unsigned char *normal_data = stbi_load("../../res/crate_1.png", &tex_w, &tex_h, &bpp, STBI_rgb_alpha);
    // if (!texture_data)
    //{
    //    fprintf(stderr, "Loading image : %s\n", stbi_failure_reason());
    //    return 0;
    //}

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
    ren_data.bpp = tex_bpp;

    // Allocate z buffer
    ren_data.z_buffer_array = (float *)malloc(sizeof(float) * (ren_data.screen_num_pixels));

    // Lights (W, Z, Y, Z)
    ren_data.light_position = _mm_set_ps(0.0f, -1.0f, 0.5f, 1.0f);
    ren_data.light_value = 0.0f;

    // Load Mesh
    Mesh_Data *mesh;
    Get_Object_Data(obj_filename, true, &mesh);

    // Projection Matrix : converts from view space to screen space
    const Mat4x4 Projection_matrix = Get_Projection_Matrix(90.0f, (float)ren.HEIGHT / (float)ren.WIDTH, 0.1f, 1000.0f);

    // Translation Matrix : Move the object in 3D space X Y Z
    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.0f, 5.0f);

    // Camera
    const __m128 camera = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

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
    while (ren.running)
    {
        // TODO : Better frame and z buffer clearing
        // SDL_Surface *screen = SDL_GetWindowSurface(window);
        // SDL_FillRect(screen, 0, 0);
        // SDL_UpdateWindowSurface(window);
        for (size_t i = 0; i < ren_data.screen_num_pixels; i++)
        {
            // ren_data.z_buffer_array[i] = 0.0f; // clear z buffer
            ren_data.pixels[i] = 0x0000; // clear screen pixels
        }

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (SDL_QUIT == event.type)
            {
                ren.running = false;
                break;
            }
        }

        Mat4x4 World_Matrix = {0.0f};
        fTheta += (float)MSPerFrame;
        // fTheta += 0.0f;

        const Mat4x4 matRotZ = Get_Rotation_Z_Matrix(fTheta); // Rotation Z
        const Mat4x4 matRotX = Get_Rotation_X_Matrix(fTheta); // Rotation X

        Matrix_Multiply_Matrix(matRotZ.elements, matRotX.elements, World_Matrix.elements);
        Matrix_Multiply_Matrix(World_Matrix.elements, Translation_matrix.elements, World_Matrix.elements);

        for (size_t i = 0; i < mesh->num_of_triangles; i += 1) // can we jump through triangles?
        {
            //__m128 tri1 = _mm_set_ps(mesh.vertex_coordinates[3 * i + 0], mesh.vertex_coordinates[3 * i + 0], mesh.vertex_coordinates[3 * i + 0], mesh.vertex_coordinates[3 * i + 0]);
            //__m128 tri2 = _mm_set_ps(mesh.vertex_coordinates[3 * i + 1], mesh.vertex_coordinates[3 * i + 1], mesh.vertex_coordinates[3 * i + 1], mesh.vertex_coordinates[3 * i + 1]);
            //__m128 tri3 = _mm_set_ps(mesh.vertex_coordinates[3 * i + 2], mesh.vertex_coordinates[3 * i + 2], mesh.vertex_coordinates[3 * i + 2], mesh.vertex_coordinates[3 * i + 2]);
            __m128 tri1 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 0]); // 12 because we load 3 triangles at at a time looping
            __m128 tri2 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 4]); // through triangles. 3 traingles each spaced 4 coordinates apart
            __m128 tri3 = _mm_load_ps(&mesh->vertex_coordinates[i * 12 + 8]); // 4 * 3 = 12; [x y z w][x y z w][x y z w]...

            // World_Matrix * Each Vertix = transformed Vertex
            tri1 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri1);
            tri2 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri2);
            tri3 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri3);

            // tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));
            // tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));
            // tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));

            // Vector Dot Product between : Surface normal and CameraRay
            const __m128 surface_normal = Calculate_Surface_Normal_SIMD(tri1, tri2, tri3);
            const __m128 camera_ray = _mm_sub_ps(tri1, camera);

            // Back face culling
            const float dot_product_result = Calculate_Dot_Product_SIMD(surface_normal, camera_ray);
            if (dot_product_result < 0)
            {
                // Illumination
                // vec3 light_direction = {0.0f, 1.0f, -1.0f};

                // How similar is normal to light direction
                const float light_value = max(0.2f, Calculate_Dot_Product_SIMD(ren_data.light_position, surface_normal));
                ren_data.light_value = light_value;
                // Convert World Space, into View Space
                // View Matrix * Each Transformed Vertex = viewed Vertex

                // 3D -> 2D
                // Matrix Projected * Viewed Vertex = projected Vertex
                tri1 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri1);
                tri2 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri2);
                tri3 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri3);

                // Setup texture coordinates
                const __m128 one_over_w1 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(tri1, tri1, _MM_SHUFFLE(3, 3, 3, 3)));
                const __m128 one_over_w2 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(tri2, tri2, _MM_SHUFFLE(3, 3, 3, 3)));
                const __m128 one_over_w3 = _mm_div_ps(_mm_set1_ps(1.0f), _mm_shuffle_ps(tri3, tri3, _MM_SHUFFLE(3, 3, 3, 3)));

                // tex coordinates are read in like : u u u...
                __m128 texture_u = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 0], mesh->uv_coordinates[6 * i + 2], mesh->uv_coordinates[6 * i + 4]);
                __m128 texture_v = _mm_set_ps(0.0f, mesh->uv_coordinates[6 * i + 1], mesh->uv_coordinates[6 * i + 3], mesh->uv_coordinates[6 * i + 5]);

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

                // Draw (CCW) Triangle Order
                // Draw_Textured_Triangle(&ren_data, tri3, tri2, tri1, texture_u, texture_v, one_over_w1, one_over_w2, one_over_w3);
                Draw_Triangle_Outline(ren_data.fmt, ren_data.pixels, tri1, tri2, tri3, &LINE_COLOUR);
            }
        }
        // Update Screen
        SDL_UpdateWindowSurface(ren.window);

        // End frame timing
        const Uint64 EndCounter = SDL_GetPerformanceCounter();
        const Uint64 CounterElapsed = EndCounter - LastCounter;

        MSPerFrame = ((double)CounterElapsed / (double)SDL_GetPerformanceFrequency());

        if (loop_counter == 50)
        {
            loop_counter = 0;
            const double FPS = (double)SDL_GetPerformanceFrequency() / (double)CounterElapsed;
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
    free(ren_data.pixels);
    free(ren_data.z_buffer_array);
    SDL_CleanUp(&ren);
    return 0;
}