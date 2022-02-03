#include "Renderer.h"

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height)
{
    Renderer rend;

    if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
    {
        fprintf(stderr, "Could not SDL_Init(SDL_INIT_VIDEO): %s\n", SDL_GetError());
        exit(2);
    }

    rend.window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_SHOWN); // show upon creation

    if (rend.window == NULL)
    {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        exit(2);
    }

    rend.renderer = SDL_CreateRenderer(rend.window, -1, SDL_RENDERER_ACCELERATED);
    if (rend.renderer == NULL)
    {
        SDL_DestroyWindow(rend.window);
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
        exit(2);
    }

    rend.running = true;
    rend.HEIGHT = height;
    rend.WIDTH = width;
    return rend;
}

void SDL_CleanUp(Renderer *renderer)
{
    SDL_DestroyRenderer(renderer->renderer);
    SDL_DestroyWindow(renderer->window);
    SDL_Quit();
}

static void Draw_Pixel(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, const SDL_Colour *col)
{
    const int index = (int)y * 1000 + (int)x;
    pixels[index] = SDL_MapRGBA(fmt,
                                (uint8_t)(col->r),
                                (uint8_t)(col->g),
                                (uint8_t)(col->b),
                                (uint8_t)(col->a));
}

static void Draw_Line(const SDL_PixelFormat *fmt, unsigned int *pixels, int x0, int y0, int x1, int y1, const SDL_Colour *col)
{
    bool steep = false;
    if (abs(x0 - x1) < abs(y0 - y1))
    { // if the line is steep, we transpose the image
        int tmp = x0;
        x0 = y0;
        y0 = tmp;

        tmp = x1;
        x1 = y1;
        y1 = tmp;

        steep = true;
    }
    if (x0 > x1)
    { // make it left−to−right
        // std::swap(x0, x1);
        int tmp = x0;
        x0 = x1;
        x1 = tmp;
        // std::swap(y0, y1);
        tmp = y0;
        y0 = y1;
        y1 = tmp;
    }
    const int dx = x1 - x0;
    const int dy = y1 - y0;
    const int derror2 = abs(dy) * 2;
    int error2 = 0;
    int y = y0;

    for (int x = x0; x <= x1; x++)
    {
        if (steep)
        {
            Draw_Pixel(fmt, pixels, y, x, col);
        }
        else
        {
            Draw_Pixel(fmt, pixels, x, y, col);
        }
        error2 += derror2;
        if (error2 > dx)
        {
            y += (y1 > y0 ? 1 : -1);
            error2 -= dx * 2;
        }
    }
}

void Draw_Triangle_Outline(const SDL_PixelFormat *fmt, unsigned int *pixels, const __m128 v1, const __m128 v2, const __m128 v3, const SDL_Colour *col)
{
    float vert1[4];
    _mm_store_ps(vert1, v1);
    float vert2[4];
    _mm_store_ps(vert2, v2);
    float vert3[4];
    _mm_store_ps(vert3, v3);

    Draw_Line(fmt, pixels, (int)vert1[0], (int)vert1[1], (int)vert2[0], (int)vert2[1], col);
    Draw_Line(fmt, pixels, (int)vert2[0], (int)vert2[1], (int)vert3[0], (int)vert3[1], col);
    Draw_Line(fmt, pixels, (int)vert3[0], (int)vert3[1], (int)vert1[0], (int)vert1[1], col);
}