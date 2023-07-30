#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <stdbool.h>
#include "utils.h"

#include "app.h"
#include "timer.h"

#include "matematika.h"

#include "SDL2/SDL.h"

// #define GRAPHICS_USE_SDL_RENDERER

typedef struct Renderer_s
{
    bool running;
    int  width;
    int  height;
    int  screen_num_pixels;

    SDL_Window *window;

#ifdef GRAPHICS_USE_SDL_RENDERER // TODO : Remove this
    SDL_Renderer *renderer;
    SDL_Texture  *texture;
#else
    SDL_Surface     *surface;
    SDL_PixelFormat *fmt;
    uint8_t         *pixels;
#endif

    float  max_depth_value;
    float *z_buffer_array;
} Renderer;

extern Renderer global_renderer;

void Reneder_Startup(const char *title, const int width, const int height);
void Renderer_Destroy(void);

inline void Renderer_Clear_ZBuffer(void)
{
    // memset(ren_data.z_buffer_array, 0xF, ren_data.screen_num_pixels * 4);
    const __m128  MAX_DEPTH = _mm_set1_ps(global_renderer.max_depth_value);
    const __m128 *END       = (__m128 *)&global_renderer.z_buffer_array[global_renderer.screen_num_pixels];

    for (__m128 *i = (__m128 *)global_renderer.z_buffer_array;
         i < END;
         i += 1)
    {
        *i = MAX_DEPTH;
    }
}

inline void Renderer_Clear_Screen_Pixels(void)
{
    memset(global_renderer.pixels, 0, global_renderer.screen_num_pixels * 4);
}

void Draw_Depth_Buffer(void);

typedef struct RasterData
{
    mvec4 screen_space_verticies[3];
    mvec4 world_space_verticies[3];
    mvec4 normals[3];
    mvec4 endpoints[3];
    mvec4 tex_u;
    mvec4 tex_v;

    float w_values[3];

    mmat3 TBN[3];
} RasterData_t;

void Flat_Shading(const RasterData_t rd[4], const uint8_t collected_triangles_count);

#endif // __RENDERER_H__