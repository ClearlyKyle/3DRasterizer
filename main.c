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
    Mesh test_square;
    test_square.count = 2;
    test_square.x = test_square_x;
    test_square.y = test_square_y;
    test_square.z = test_square_z;

    // Projection Matrix : converts from view space to screen space
    const Mat4x4 matProj = Get_Projection_Matrix(90.0f, (float)ren.HEIGHT / (float)ren.WIDTH, 0.1f, 1000.0f);

    const Mat4x4 Translation_matrix = Get_Translation_Matrix(0.0f, 0.0f, 5.0f);

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

        const Mat4x4 matRotZ = Get_Rotation_Z_Matrix(fTheta); // Rotation Z
        const Mat4x4 matRotX = Get_Rotation_X_Matrix(fTheta); // Rotation X

        Matrix_Multiply_Matrix(matRotZ.elements, matRotX.elements, World_Matrix.elements);
        Matrix_Multiply_Matrix(World_Matrix.elements, Translation_matrix.elements, World_Matrix.elements);

        for (size_t i = 0; i < test_square.count; i += 3)
        {
            /* code */
            Matrix_Multiply_Matrix(World_Matrix.elements, test_square.x);
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