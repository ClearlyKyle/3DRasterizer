#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "SDL2/SDL.h"

#include "vector.h"
#include "Renderer.h"

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 1000
const unsigned int num_of_pixels = SCREEN_WIDTH * SCREEN_HEIGHT;

int main(int argc, char *argv[])
{
    Renderer ren = SDL_Startup("Window", SCREEN_HEIGHT, SCREEN_WIDTH);

    // ren.buffer = (float *)malloc(sizeof(float) * (num_of_pixels));
    ren.ZBuffer = (float *)malloc(sizeof(float) * (num_of_pixels));

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
        // clear the z buffer
        for (size_t i = 0; i < num_of_pixels; i++)
        {
            ren.ZBuffer[i] = FLT_MAX;
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

        // Update Screen
        SDL_RenderPresent(ren.renderer);

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