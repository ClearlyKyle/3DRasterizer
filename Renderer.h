#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <stdbool.h>

#include "vector.h"
#include "lights.h"

#include "SDL2/SDL.h"

typedef struct Renderer_s
{
    unsigned int WIDTH;
    unsigned int HEIGHT;
    bool running;

    SDL_Window *window;
    SDL_Renderer *renderer;
} Renderer;

typedef struct Rendering_data_s
{
    SDL_Surface *surface;
    SDL_PixelFormat *fmt;
    unsigned int *pixels;
    unsigned int screen_width;
    unsigned int screen_height;
    unsigned int screen_num_pixels;

    float max_depth_value;
    float *z_buffer_array;

    unsigned char *tex_data;
    unsigned int tex_w;
    unsigned int tex_h;
    unsigned int tex_bpp;

    unsigned char *nrm_data;
    unsigned int nrm_w;
    unsigned int nrm_h;
    unsigned int nrm_bpp;

    __m128 light_position;
    float light_value;

    Shading_Mode shading;

} Rendering_data;

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height);
void SDL_CleanUp(Renderer *renderer);

void Draw_Triangle_Outline(const Rendering_data *ren, const __m128 *verticies, const SDL_Colour *col);

void Draw_Depth_Buffer(const Rendering_data *render_data);

void Textured_Shading(const Rendering_data *render, const __m128 *screen_space, const __m128 *world_space,
                      const __m128 *w_values, const __m128 *normal_values, const __m128 texture_u, const __m128 texture_v,
                      const __m128 surface_normal, const PointLight *light, const Mat4x4 TBN);

void Flat_Shading(const Rendering_data *render, const __m128 *screen_space, const __m128 *world_space, const __m128 *w_values, const __m128 *normal_values,
                  const __m128 surface_normal, const PointLight *light, const Shading_Mode shading);

#endif // __RENDERER_H__