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

// THE EXTREMELY FAST LINE ALGORITHM Variation E (Addition Fixed Point PreCalc)
// http://www.edepot.com/algorithm.html
static void DrawLine2(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, int x2, int y2, const SDL_Colour *col)
{
    bool yLonger = false;
    int shortLen = y2 - y;
    int longLen = x2 - x;
    if (abs(shortLen) > abs(longLen))
    {
        int swap = shortLen;
        shortLen = longLen;
        longLen = swap;
        yLonger = true;
    }
    int decInc;
    if (longLen == 0)
        decInc = 0;
    else
        decInc = (shortLen << 16) / longLen;

    if (yLonger)
    {
        if (longLen > 0)
        {
            longLen += y;
            for (int j = 0x8000 + (x << 16); y <= longLen; ++y)
            {
                // myPixel(surface, j >> 16, y);
                Draw_Pixel(fmt, pixels, j >> 16, y, col);
                j += decInc;
            }
            return;
        }
        longLen += y;
        for (int j = 0x8000 + (x << 16); y >= longLen; --y)
        {
            // myPixel(surface, j >> 16, y);
            Draw_Pixel(fmt, pixels, j >> 16, y, col);

            j -= decInc;
        }
        return;
    }

    if (longLen > 0)
    {
        longLen += x;
        for (int j = 0x8000 + (y << 16); x <= longLen; ++x)
        {
            // myPixel(surface, x, j >> 16);
            Draw_Pixel(fmt, pixels, x, j >> 16, col);

            j += decInc;
        }
        return;
    }
    longLen += x;
    for (int j = 0x8000 + (y << 16); x >= longLen; --x)
    {
        // myPixel(surface, x, j >> 16);
        Draw_Pixel(fmt, pixels, x, j >> 16, col);
        j -= decInc;
    }
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

    // Draw_Line(fmt, pixels, (int)vert1[0], (int)vert1[1], (int)vert2[0], (int)vert2[1], col);
    // Draw_Line(fmt, pixels, (int)vert2[0], (int)vert2[1], (int)vert3[0], (int)vert3[1], col);
    // Draw_Line(fmt, pixels, (int)vert3[0], (int)vert3[1], (int)vert1[0], (int)vert1[1], col);
    DrawLine2(fmt, pixels, (int)vert1[0], (int)vert1[1], (int)vert2[0], (int)vert2[1], col);
    DrawLine2(fmt, pixels, (int)vert2[0], (int)vert2[1], (int)vert3[0], (int)vert3[1], col);
    DrawLine2(fmt, pixels, (int)vert3[0], (int)vert3[1], (int)vert1[0], (int)vert1[1], col);
}

static __m128 Get_AABB_SIMD(const __m128 v1, const __m128 v2, const __m128 v3)
{
    const __m128 max_values = _mm_min_ps(_mm_max_ps(_mm_max_ps(v1, v2), v3), _mm_set_ps(0.0f, 0.0f, 100.0f, 100.0f));
    const __m128 min_values = _mm_max_ps(_mm_min_ps(_mm_min_ps(v1, v2), v3), _mm_set1_ps(0.0f));

    // Returns {maxX, minX, maxY, minY}
    return _mm_unpacklo_ps(max_values, min_values);
}

// static vec3 calculate_weights_consts(vec3 ab, vec3 ac, vec3 ap)
static __m128 SIMD_Get_Weights(const __m128 a, const __m128 b, const __m128 c, const __m128 p)
{
    // first 3 are const
    const __m128 ab = _mm_sub_ps(b, a);
    const __m128 ac = _mm_sub_ps(c, a);
    const __m128 cb = _mm_sub_ps(b, c);

    const __m128 ap = _mm_sub_ps(p, a);
    const __m128 cp = _mm_sub_ps(p, c);

    //__declspec(align(16)) float tmp[4];

    // 1.0f / (ab.x * ac.y - ab.y * ac.x);
    ///////// (ab.x * ap.y - ab.y * ap.x) * factor;
    ///////// (ap.x * ac.y - ap.y * ac.x) * factor;
    ///////// (cb.y * cp.y - cb.x * cp.x) * factor;

    const __m128 r1 = _mm_set_ps(cb[1], ap[0], ab[0], ab[0]);
    const __m128 r2 = _mm_set_ps(cp[1], ac[1], ap[1], ac[1]);
    const __m128 r3 = _mm_set_ps(cb[0], ap[1], ab[1], ab[1]);
    const __m128 r4 = _mm_set_ps(cp[0], ac[0], ap[0], ac[0]);

    const __m128 res = _mm_sub_ps(
        _mm_mul_ps(r1, r2),
        _mm_mul_ps(r3, r4));

    // const __m128 factor = _mm_rcp_ps(_mm_set1_ps(res[0]));
    const __m128 factor = _mm_div_ps(_mm_set1_ps(1.0f), _mm_set1_ps(res[0]));
    const __m128 weights = _mm_mul_ps(factor, res);

    return weights;
}

static vec3 calculate_weights_consts(vec3 ab, vec3 ac, vec3 ap)
{

    const float factor = 1.0f / (ab.x * ac.y - ab.y * ac.x);
    const float s = (ac.y * ap.x - ac.x * ap.y) * factor;
    const float t = (ab.x * ap.y - ab.y * ap.x) * factor;

    return (vec3){1.0f - s - t, s, t, 1.0f};
}

void Barycentric_Algorithm_Tex_Buffer(const Rendering_data *render, const __m128 v1, const __m128 v2, const __m128 v3)
// void Barycentric_Algorithm_Tex_Buffer(const SDL_PixelFormat *fmt, unsigned int *pixels, float *z_buffer_array, unsigned char *tex_data, const Triangle *tri, const Triangle *tex)
{
    /* get the bounding box of the triangle */
    float AABB_values[4]; // {maxX, minX, maxY, minY}
    _mm_store_ps(AABB_values, Get_AABB_SIMD(v1, v2, v3));

    // constants for the weights function
    const __m128 ab = _mm_sub_ps(v1, v2);
    const __m128 ac = _mm_sub_ps(v1, v3);

    // bool outside_triangle = true;

    for (int y = AABB_values[3]; y <= AABB_values[2]; y++)
    {
        for (int x = AABB_values[1]; x <= AABB_values[0]; x++)
        {
            const __m128 point = _mm_set_ps(1.0f, 1.0f, (float)y + 0.5f, (float)x + 0.5f);

            const __m128 weights = SIMD_Get_Weights(v1, v2, v3, point);

            if (weights.x > 0 && weights.y > 0 && weights.z > 0)
            { /* inside triangle */
                // outside_triangle = false;

                // Depth interpolation
                // const float z = 1.0f / ((tex->vec[0].w * weights.x) + (v1.w * weights.y) + (v2.w * weights.z));
                const float z = 1.0f / ((tex->vec[0].w * weights.x) + (tex->vec[1].w * weights.y) + (tex->vec[2].w * weights.z));
                const float z = 1.0f / ((tex->vec[0].w * weights.x) + (tex->vec[1].w * weights.y) + (tex->vec[2].w * weights.z));

                // Get z-buffer index
                const int index = (int)y * 1000 + (int)x;

                if (z < z_buffer_array[index])
                {
                    // Set new value in zbuffer if point is closer
                    z_buffer_array[index] = z;

                    //  IMAGE TEXTURE
                    const float u = (320 - 1) * (weights.x * tex->vec[0].x + weights.y * tex->vec[1].x + weights.z * tex->vec[2].x) * z;
                    const float v = (320 - 1) * (weights.x * tex->vec[0].y + weights.y * tex->vec[1].y + weights.z * tex->vec[2].y) * z;

                    const unsigned char *pixelOffset = tex_data + ((int)v + 320 * (int)u) * 4; // 4 = bpp
                    // const SDL_Colour colour = {.r = (uint8_t)(pixelOffset[0]),
                    //                            .g = (uint8_t)(pixelOffset[1]),
                    //                            .b = (uint8_t)(pixelOffset[2]),
                    //                            .a = (uint8_t)(pixelOffset[3])};

                    // Pack these RGBA values into a pixel of the correct format.
                    pixels[index] = SDL_MapRGBA(fmt,
                                                (uint8_t)(pixelOffset[0]),
                                                (uint8_t)(pixelOffset[1]),
                                                (uint8_t)(pixelOffset[2]),
                                                (uint8_t)(pixelOffset[3]));

                    // CHECKERBOARD PATTERN
                    // tex_data = NULL;
                    // const float u = ((weights.x * texture1.u + weights.y * texture2.u + weights.z * texture3.u) * z);
                    // const float v = ((weights.x * texture1.v + weights.y * texture2.v + weights.z * texture3.v) * z);
                    // const float M = 8.0f;
                    // const float p = (float)((fmod(u * M, 1.0) > 0.5) ^ (fmod(v * M, 1.0) < 0.5));
                    // const SDL_Colour colour = {.r = (uint8_t)((p * 255)),
                    //                           .g = (uint8_t)((p * 255)),
                    //                           .b = (uint8_t)((p * 255)),
                    //                           .a = (uint8_t)255};

                    // COLOURS
                    // const float r = (weights.x * c0.x + weights.y * c0.y + weights.z * c0.z) * z;
                    // const float g = (weights.x * c1.x + weights.y * c1.y + weights.z * c1.z) * z;
                    // const float b = (weights.x * c2.x + weights.y * c2.y + weights.z * c2.z) * z;
                    // const SDL_Colour colour = {(uint8_t)(r * 255), (uint8_t)(g * 255), (uint8_t)(b * 255), 255};

                    // Draw the pixel finally
                    // Draw_Point(renderer, x, y, &colour);
                }
            }
            // else
            //{
            //     if (outside_triangle == false)
            //     {
            //         outside_triangle = true;
            //         break;
            //     }
            // }
        }
    }
}