#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "SDL2/SDL.h"

#include "vector.h"
#include "Renderer.h"
#include "test_sqaure.h"

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 1000

int main(int argc, char *argv[])
{
    Renderer ren = SDL_Startup("Window", SCREEN_HEIGHT, SCREEN_WIDTH);

    SDL_Surface *window_surface = SDL_GetWindowSurface(ren.window);

    // Get window pixels
    unsigned int *pixels = window_surface->pixels;
    const int width = window_surface->w;
    const int height = window_surface->h;
    const int number_of_pixels = width * height;

    // Allocate z buffer
    ren.ZBuffer = (float *)malloc(sizeof(float) * (number_of_pixels));

    // Test Square
    Mesh test_square = {
        .x = test_square_x,
        .y = test_square_y,
        .z = test_square_z,
        .w = test_square_w,
        .count = 12};
    // test_square.count = 2;
    // test_square.x = test_square_x;
    // test_square.y = test_square_y;
    // test_square.z = test_square_z;

    // Projection Matrix : converts from view space to screen space
    const Mat4x4 Projection_matrix = Get_Projection_Matrix(90.0f, (float)ren.HEIGHT / (float)ren.WIDTH, 0.1f, 1000.0f);

    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.0f, 2.0f);

    // Camera
    __m128 camera = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

    // Performance counters
    Uint64 LastCounter = SDL_GetPerformanceCounter();
    double MSPerFrame = 0.0;
    float fTheta = 0.0f;
    char window_title[32];

    SDL_Event event;
    ren.running = true;
    while (ren.running)
    {
        for (size_t i = 0; i < number_of_pixels; i++)
        {
            ren.ZBuffer[i] = FLT_MAX; // clear z buffer
            pixels[i] = 0x0000;       // clear screen pixels
        }

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

        for (size_t i = 0; i < test_square.count; i++)
        {
            //__m128 tri1 = _mm_set_ps(test_square.w[3 * i], test_square.z[3 * i], test_square.y[3 * i], test_square.x[3 * i]);
            //__m128 tri2 = _mm_set_ps(test_square.w[3 * i + 1], test_square.z[3 * i + 1], test_square.y[3 * i + 1], test_square.x[3 * i + 1]);
            //__m128 tri3 = _mm_set_ps(test_square.w[3 * i + 2], test_square.z[3 * i + 2], test_square.y[3 * i + 2], test_square.x[3 * i + 2]);
            __m128 tri1 = _mm_load_ps(&test_contig_data[12 * i + 0]);
            __m128 tri2 = _mm_load_ps(&test_contig_data[12 * i + 4]);
            __m128 tri3 = _mm_load_ps(&test_contig_data[12 * i + 8]);

            // World_Matrix * Each Vertix = transformed Vertex
            tri1 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri1);
            tri2 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri2);
            tri3 = Matrix_Multiply_Vector_SIMD(World_Matrix.elements, tri3);

            tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));
            tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));
            tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 5.0f, 0.0f, 0.0f));

            // Vector Dot Product between : Surface normal and CameraRay
            const __m128 surface_normal = Calculate_Surface_Normal_SIMD(tri1, tri2, tri3);
            const __m128 camera_ray = _mm_sub_ps(tri1, camera);

            const float dot_product_result = Calculate_Dot_Product_SIMD(surface_normal, camera_ray);
            if (dot_product_result < 0)
            {
                // Convert World Space, into View Space
                // View Matrix * Each Transformed Vertex = viewed Vertex

                // Back face culling

                // 3D -> 2D
                // Matrix Projected * Viewed Vertex = projected Vertex
                tri1 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri1);
                tri2 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri2);
                tri3 = Matrix_Multiply_Vector_SIMD(Projection_matrix.elements, tri3);

                // Perform x/w, y/w, z/w
                // __m128 _mm_rcp_ps (__m128 a)
                tri1 = _mm_div_ps(tri1, _mm_shuffle_ps(tri1, tri1, _MM_SHUFFLE(3, 3, 3, 3)));
                tri2 = _mm_div_ps(tri2, _mm_shuffle_ps(tri2, tri2, _MM_SHUFFLE(3, 3, 3, 3)));
                tri3 = _mm_div_ps(tri3, _mm_shuffle_ps(tri3, tri3, _MM_SHUFFLE(3, 3, 3, 3)));

                // Sacle Into View
                tri1 = _mm_add_ps(tri1, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
                tri2 = _mm_add_ps(tri2, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));
                tri3 = _mm_add_ps(tri3, _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f));

                const float x_adjustment = 0.5f * (float)ren.WIDTH;
                const float y_adjustment = 0.5f * (float)ren.HEIGHT;

                tri1 = _mm_mul_ps(tri1, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
                tri2 = _mm_mul_ps(tri2, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));
                tri3 = _mm_mul_ps(tri3, _mm_set_ps(1.0f, 1.0f, y_adjustment, x_adjustment));

                // Draw
                const SDL_Colour WHITE = {255, 255, 255, 255};
                Draw_Triangle_Outline(window_surface->format, pixels, tri1, tri2, tri3, &WHITE);
            }
        }
        // Update Screen
        SDL_UpdateWindowSurface(ren.window);

        // End frame timing
        const Uint64 EndCounter = SDL_GetPerformanceCounter();
        const Uint64 CounterElapsed = EndCounter - LastCounter;

        MSPerFrame = ((double)CounterElapsed / (double)SDL_GetPerformanceFrequency());
        const double FPS = (double)SDL_GetPerformanceFrequency() / (double)CounterElapsed;
        snprintf(window_title, sizeof(window_title), "%.02f ms/f \t%.02f f/s\n", 1000.0 * MSPerFrame, FPS);

        SDL_SetWindowTitle(ren.window, window_title);

        LastCounter = EndCounter;
    }

    SDL_CleanUp(&ren);
    return 0;
}