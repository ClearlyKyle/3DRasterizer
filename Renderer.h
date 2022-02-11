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
    unsigned int screen_width;
    unsigned int screen_height;

    float *z_buffer_array;

    unsigned char *tex_data;
    unsigned int tex_w;
    unsigned int tex_h;
    unsigned int bpp;
} Rendering_data;

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height);
void SDL_CleanUp(Renderer *renderer);

void Draw_Triangle_Outline(const Rendering_data *render, unsigned int *pixels, const __m128 v1, const __m128 v2, const __m128 v3, const SDL_Colour *col);
void Draw_Textured_Triangle(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2, const __m128 texture_u, const __m128 texture_v);

#endif // __RENDERER_H__