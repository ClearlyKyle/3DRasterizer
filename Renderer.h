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

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height);
void SDL_CleanUp(Renderer *renderer);

#endif // __RENDERER_H__