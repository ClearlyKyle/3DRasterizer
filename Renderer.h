#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <stdbool.h>

#include "textures.h"
#include "AABB.h"
#include "vector.h"
#include "lights.h"
#include "timer.h"

#include "SDL2/SDL.h"

typedef struct Renderer_s
{
    bool running;
    int  width;
    int  height;
    int  screen_num_pixels;

    SDL_Window  *window;
    SDL_Surface *surface;

    SDL_PixelFormat *fmt;
    unsigned int    *pixels;

    float  max_depth_value;
    float *z_buffer_array;
} Renderer;

// App state and make it global
// Add camera position
typedef struct AppState_s
{
    Texture tex; // The diffuse texture
    Texture nrm; // The normal map texture

    __m128 camera_position;

    // Shading_Mode shading;
    Shading_Mode shading_mode;
} AppState;

// TODO: Just make this global
extern Renderer global_renderer;
extern AppState global_app;

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
void Draw_Triangle_Outline(const __m128 *verticies, const SDL_Colour col);

void Flat_Shading(const __m128 *screen_space_verticies, const __m128 *world_space_verticies, const float *w_values, const __m128 *normal_values, const Light *light);

void Textured_Shading(const __m128 *screen_space_verticies, const __m128 *world_space_verticies, const float *w_values, const __m128 *normal_values,
                      const __m128 texture_u, const __m128 texture_v, const Mat3x3 TBN, const Light *light);

#endif // __RENDERER_H__