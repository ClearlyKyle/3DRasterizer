#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#include "SDL2/SDL.h"

#include "lights.h"
#include "vector.h"
#include "Renderer.h"
#include "test_sqaure.h"
#include "ObjLoader.h"

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 900

int main(int argc, char *argv[])
{
    Renderer ren = SDL_Startup("Window", SCREEN_WIDTH, SCREEN_HEIGHT);

    SDL_Surface *window_surface = SDL_GetWindowSurface(ren.window);

    // WOODEN BOX
    // const char *obj_filename = "../../res/Wooden Box/wooden crate.obj";
    // const char *tex_filename = "../../res/Wooden Box/crate_BaseColor.png";
    const char *nrm_filename = "../../res/Wooden Box/crate_Normal.png";

    // CRATE
    // const char *obj_filename = "../../res/Crate/cube_triang.obj";
    // const char *obj_filename = "../../res/Crate/Crate1.obj";
    const char *tex_filename = "../../res/Crate/crate_1.png";

    // SPHERE
    const char *obj_filename = "../../res/Sphere/sphere.obj";

    // Load texture data
    int tex_w, tex_h, tex_bpp;
    // unsigned char *texture_data = stbi_load(tex_filename, &tex_w, &tex_h, &tex_bpp, STBI_rgb);
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
    int nrm_w, nrm_h, nrm_bpp;
    unsigned char *normal_data = stbi_load(nrm_filename, &nrm_w, &nrm_h, &nrm_bpp, STBI_rgb_alpha);
    if (!texture_data)
    {
        fprintf(stderr, "Loading image : %s\n", stbi_failure_reason());
        return 0;
    }

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
    ren_data.z_buffer_array = (unsigned int *)z_buff;

    // Lights (W, Z, Y, X)
    ren_data.light_position = _mm_set_ps(0.0f, -1.0f, 0.0f, 1.0f);
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

    // Lights -------------------------
    Fragment frag;
    frag.color = _mm_set1_ps(1.0f);
    frag.diffuse = _mm_load_ps(mesh->diffuse);
    // frag.ambient = _mm_load_ps(mesh->ambient);
    frag.ambient = _mm_set1_ps(0.5f);
    frag.specular = _mm_load_ps(mesh->specular);
    // const __m128 light_position = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);

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
        // Clear Z Buffer
        memset(ren_data.z_buffer_array, 0, ren_data.screen_num_pixels * 4);
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

        Mat4x4 World_Matrix = {0.0f};
        fTheta += (float)MSPerFrame;
        // fTheta += 0.0f;

        const Mat4x4 matRotZ = Get_Rotation_Z_Matrix(fTheta); // Rotation Z
        const Mat4x4 matRotX = Get_Rotation_X_Matrix(fTheta); // Rotation X

        Matrix_Multiply_Matrix(matRotZ.elements, matRotX.elements, World_Matrix.elements);
        Matrix_Multiply_Matrix(World_Matrix.elements, Translation_matrix.elements, World_Matrix.elements);

        for (size_t i = 0; i < mesh->num_of_triangles; i += 1) // can we jump through triangles?
        {
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

                // AMBIENT
                const __m128 ambient = _mm_mul_ps(frag.color, frag.ambient);

                // DIFFUSE
                const __m128 norm = Normalize_m128(surface_normal);
                // const __m128 lightDir = _mm_sub_ps(ren_data.light_position, tri1);
                const float diff = (float)fmax((double)Calculate_Dot_Product_SIMD(ren_data.light_position, norm), 0.0);
                const __m128 diffuse = _mm_mul_ps(frag.diffuse, _mm_set1_ps(diff));

                // SPECULAR
                // vec3 viewDir = normalize(viewPos - FragPos);
                const __m128 reflectDir = Reflect_m128(_mm_mul_ps(ren_data.light_position, _mm_set1_ps(-1.0f)), norm); // I - 2.0 * dot(N, I) * N

                const float spec = (float)pow(fmax(Calculate_Dot_Product_SIMD(camera, reflectDir), 0.0), 32);
                const __m128 specular = _mm_mul_ps(_mm_mul_ps(frag.color, _mm_set1_ps(spec)), frag.specular);

                const __m128 frag_colour = Clamp_m128(_mm_add_ps(ambient, diffuse), 0.0f, 1.0f);
                // const __m128 frag_colour = Clamp_m128(_mm_add_ps(_mm_add_ps(frag.ambient, diffuse), specular), 0.0f, 1.0f);

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

                //__m128 edge1 = _mm_add_ps(tri2, tri1);
                //__m128 edge2 = _mm_add_ps(tri3, tri1);
                //__m128 deltaUV1 = uv2 - uv1;
                //__m128 deltaUV2 = uv3 - uv1;

                // float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

                // tangent1.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
                // tangent1.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
                // tangent1.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

                // bitangent1.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
                // bitangent1.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
                // bitangent1.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

                // similar procedure for calculating tangent/bitangent for plane's second triangle

                // Draw (CCW) Triangle Order
                // Draw_Textured_Triangle(&ren_data, tri3, tri2, tri1, texture_u, texture_v, one_over_w1, one_over_w2, one_over_w3, frag_colour);
                Draw_Triangle_Outline(ren_data.fmt, ren_data.pixels, tri1, tri2, tri3, &LINE_COLOUR);
            }
        }
        // Update Screen
        SDL_UpdateWindowSurface(ren.window);

        // End frame timing
        const Uint64 EndCounter = SDL_GetPerformanceCounter();
        const Uint64 CounterElapsed = EndCounter - LastCounter;

        MSPerFrame = ((double)CounterElapsed / (double)SDL_GetPerformanceFrequency());

        if (loop_counter == 32)
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
    // free(ren_data.pixels);
    // free(ren_data.z_buffer_array);
    SDL_CleanUp(&ren);
    return 0;
}