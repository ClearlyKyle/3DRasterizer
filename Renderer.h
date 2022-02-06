#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <stdbool.h>

#include "SDL2/SDL.h"

typedef struct Renderer_s
{
    unsigned int WIDTH;
    unsigned int HEIGHT;
    bool running;

    float *ZBuffer;
    float *buffer;

    SDL_Window *window;
    SDL_Renderer *renderer;
} Renderer;

typedef struct Rendering_data_s
{
    const SDL_PixelFormat *fmt;
    unsigned int *pixels;
    float *z_buffer_array;
    unsigned char *tex_data;
} Rendering_data;

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height);
void SDL_CleanUp(Renderer *renderer);

void Draw_Triangle_Outline(const SDL_PixelFormat *fmt, unsigned int *pixels, const __m128 v1, const __m128 v2, const __m128 v3, const SDL_Colour *col);
void Barycentric_Algorithm_Tex_Buffer(const Rendering_data *render, const __m128 v1, const __m128 v2, const __m128 v3);

#endif // __RENDERER_H__