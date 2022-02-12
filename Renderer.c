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

static void Draw_Pixel_SDL_Colour(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, const SDL_Colour *col)
{
    // index = y * screen_w * x
    const int index = (int)y * 1000 + (int)x;
    pixels[index] = SDL_MapRGBA(fmt,
                                (uint8_t)(col->r),
                                (uint8_t)(col->g),
                                (uint8_t)(col->b),
                                (uint8_t)(col->a));
}
static void Draw_Pixel_RGBA(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
{
    // index = y * screen_w * x
    const int index = (int)y * 1000 + (int)x;
    pixels[index] = SDL_MapRGBA(fmt,
                                red,
                                green,
                                blue,
                                alpha);
}

static void Draw_Pixel_Pixel_Data(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, const unsigned char *texture_data)
{
    // index = y * screen_w * x
    const int index = (int)y * 1000 + (int)x;
    pixels[index] = SDL_MapRGBA(fmt,
                                (uint8_t)(texture_data[0]),
                                (uint8_t)(texture_data[1]),
                                (uint8_t)(texture_data[2]),
                                (uint8_t)(texture_data[3]));
}

// THE EXTREMELY FAST LINE ALGORITHM Variation E (Addition Fixed Point PreCalc)
// http://www.edepot.com/algorithm.html
static void Draw_Line(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, int x2, int y2, const SDL_Colour *col)
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
                Draw_Pixel_SDL_Colour(fmt, pixels, j >> 16, y, col);
                j += decInc;
            }
            return;
        }
        longLen += y;
        for (int j = 0x8000 + (x << 16); y >= longLen; --y)
        {
            // myPixel(surface, j >> 16, y);
            Draw_Pixel_SDL_Colour(fmt, pixels, j >> 16, y, col);

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
            Draw_Pixel_SDL_Colour(fmt, pixels, x, j >> 16, col);

            j += decInc;
        }
        return;
    }
    longLen += x;
    for (int j = 0x8000 + (y << 16); x >= longLen; --x)
    {
        // myPixel(surface, x, j >> 16);
        Draw_Pixel_SDL_Colour(fmt, pixels, x, j >> 16, col);
        j -= decInc;
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

union AABB_u
{
    struct
    {
        int maxX;
        int minX;
        int maxY;
        int minY;
    };
    int values[4];
};

static __m128i Get_AABB_SIMD(const __m128 v1, const __m128 v2, const __m128 v3, int screen_width, int screen_height)
{
    const __m128i max_values = _mm_cvtps_epi32(_mm_min_ps(_mm_max_ps(_mm_max_ps(v1, v2), v3), _mm_set_ps(0.0f, 0.0f, (float)screen_width, (float)screen_height)));
    const __m128i min_values = _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(_mm_min_ps(v1, v2), v3), _mm_set1_ps(0.0f)));

    // Returns {maxX, minX, maxY, minY}
    return _mm_unpacklo_epi32(max_values, min_values);
}

// https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
void Draw_Textured_Triangle(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2,
                            const __m128 texture_u, const __m128 texture_v,
                            const __m128 one_over_w1, const __m128 one_over_w2, const __m128 one_over_w3)
{
    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128 v0_x = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 v1_x = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 v2_x = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0));

    const __m128 v0_y = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 v1_y = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 v2_y = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1));

    // Actually the W value, shhh!
    // const __m128 v0_z = _mm_shuffle_ps(one_over_w1, one_over_w1, _MM_SHUFFLE(3, 3, 3, 3));
    // const __m128 v1_z = _mm_shuffle_ps(one_over_w2, one_over_w2, _MM_SHUFFLE(3, 3, 3, 3));
    // const __m128 v2_z = _mm_shuffle_ps(one_over_w3, one_over_w3, _MM_SHUFFLE(3, 3, 3, 3));

    // Edge Setup
    const __m128 A01 = _mm_sub_ps(v0_y, v1_y); // A01 = (int)(v0.y - v1.y);
    const __m128 A12 = _mm_sub_ps(v1_y, v2_y); // A12 = (int)(v1.y - v2.y);
    const __m128 A20 = _mm_sub_ps(v2_y, v0_y); // A20 = (int)(v2.y - v0.y);

    const __m128 B01 = _mm_sub_ps(v1_x, v0_x);
    const __m128 B12 = _mm_sub_ps(v2_x, v1_x);
    const __m128 B20 = _mm_sub_ps(v0_x, v2_x);

    const __m128 C01 = _mm_sub_ps(_mm_mul_ps(v0_x, v1_y), _mm_mul_ps(v0_y, v1_x));
    const __m128 C12 = _mm_sub_ps(_mm_mul_ps(v1_x, v2_y), _mm_mul_ps(v1_y, v2_x));
    const __m128 C20 = _mm_sub_ps(_mm_mul_ps(v2_x, v0_y), _mm_mul_ps(v2_y, v0_x));

    const __m128 p_x = _mm_add_ps(_mm_set1_ps((float)aabb.minX), _mm_set_ps(0, 1, 2, 3));
    const __m128 p_y = _mm_set1_ps((float)aabb.minY);

    // Barycentric Setip
    // Order of triangle sides *IMPORTANT*
    // v1, v2 :  w0_row = (A12 * p.x) + (B12 * p.y) + C12;
    // v2, v0 :  w1_row = (A20 * p.x) + (B20 * p.y) + C20;
    // v0, v1 :  w2_row = (A01 * p.x) + (B01 * p.y) + C01;
    __m128i w0_row = _mm_cvtps_epi32(
        _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(A12, p_x), _mm_mul_ps(B12, p_y)),
            C12));
    __m128i w1_row = _mm_cvtps_epi32(
        _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(A20, p_x), _mm_mul_ps(B20, p_y)),
            C20));
    __m128i w2_row = _mm_cvtps_epi32(
        _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(A01, p_x), _mm_mul_ps(B01, p_y)),
            C01));

    // Compute triangle area
    __m128 triArea = _mm_mul_ps(A01, v2_x);
    triArea = _mm_add_ps(triArea, _mm_mul_ps(B01, v2_y));
    triArea = _mm_add_ps(triArea, C01);

    const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), triArea);

    // X Step
    const __m128i X_Step_w0 = _mm_mullo_epi32(_mm_cvtps_epi32(A12), _mm_set1_epi32(4));
    const __m128i X_Step_w1 = _mm_mullo_epi32(_mm_cvtps_epi32(A20), _mm_set1_epi32(4));
    const __m128i X_Step_w2 = _mm_mullo_epi32(_mm_cvtps_epi32(A01), _mm_set1_epi32(4));
    // Y Step
    // const __m128i Y_Step = 1;s

    // Rasterize
    for (int y = aabb.minY; y <= aabb.maxY; y += 1)
    {
        // Barycentric coordinates at start of row
        __m128i w0 = w0_row;
        __m128i w1 = w1_row;
        __m128i w2 = w2_row;

        for (int x = aabb.minX; x <= aabb.maxX; x += 4,
                 w0 = _mm_add_epi32(w0, X_Step_w0),
                 w1 = _mm_add_epi32(w1, X_Step_w1),
                 w2 = _mm_add_epi32(w2, X_Step_w2))
        // One step to the right
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(w0, w1), w2));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(w0), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(w1), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(w2), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(w0_area, one_over_w3);
            depth = _mm_add_ps(depth, _mm_mul_ps(w1_area, one_over_w2));
            depth = _mm_add_ps(depth, _mm_mul_ps(w2_area, one_over_w1));
            depth = _mm_div_ps(_mm_set1_ps(1.0f), depth);

            const int z_index = x + render->screen_width * y;

            __m128 previousDepthValue = _mm_load_ps(&render->z_buffer_array[z_index]);
            previousDepthValue = _mm_shuffle_ps(previousDepthValue, previousDepthValue, _MM_SHUFFLE(0, 1, 2, 3));
            const __m128 depthMask = _mm_cmpge_ps(depth, previousDepthValue); // dst[i+31:i] := ( a[i+31:i] >= b[i+31:i] ) ? 0xFFFFFFFF : 0

            // mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(0, 1, 2, 3)); // reverse the mask
            const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(depthMask));

            if (finalMask.m128i_i32[3])
            // if (mask.m128i_i32[3] && finalMask.m128i_i32[3])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[3], w1_area.m128_f32[3], w0_area.m128_f32[3]);
                __m128 u = _mm_mul_ps(texture_u, weights);
                u = _mm_mul_ps(u, _mm_set1_ps(depth.m128_f32[3]));
                u = _mm_mul_ps(u, _mm_set1_ps((float)render->tex_w));

                __m128 v = _mm_mul_ps(texture_v, weights);
                v = _mm_mul_ps(v, _mm_set1_ps(depth.m128_f32[3]));
                v = _mm_mul_ps(v, _mm_set1_ps((float)render->tex_h));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->bpp;
                Draw_Pixel_Pixel_Data(render->fmt, render->pixels, x + 0, y, pixelOffset);
            }

            if (finalMask.m128i_i32[2])
            // if (mask.m128i_i32[2] && finalMask.m128i_i32[2])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[2], w1_area.m128_f32[2], w0_area.m128_f32[2]);
                __m128 u = _mm_mul_ps(texture_u, weights);
                u = _mm_mul_ps(u, _mm_set1_ps(depth.m128_f32[2]));
                u = _mm_mul_ps(u, _mm_set1_ps((float)render->tex_w));

                __m128 v = _mm_mul_ps(texture_v, weights);
                v = _mm_mul_ps(v, _mm_set1_ps(depth.m128_f32[2]));
                v = _mm_mul_ps(v, _mm_set1_ps((float)render->tex_h));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->bpp;
                Draw_Pixel_Pixel_Data(render->fmt, render->pixels, x + 1, y, pixelOffset);
            }

            if (finalMask.m128i_i32[1])
            // if (mask.m128i_i32[1] && finalMask.m128i_i32[1])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[1], w1_area.m128_f32[1], w0_area.m128_f32[1]);
                __m128 u = _mm_mul_ps(texture_u, weights);
                u = _mm_mul_ps(u, _mm_set1_ps(depth.m128_f32[1]));
                u = _mm_mul_ps(u, _mm_set1_ps((float)render->tex_w));

                __m128 v = _mm_mul_ps(texture_v, weights);
                v = _mm_mul_ps(v, _mm_set1_ps(depth.m128_f32[1]));
                v = _mm_mul_ps(v, _mm_set1_ps((float)render->tex_h));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->bpp;
                Draw_Pixel_Pixel_Data(render->fmt, render->pixels, x + 2, y, pixelOffset);
            }

            if (finalMask.m128i_i32[0])
            // if (mask.m128i_i32[0] && finalMask.m128i_i32[0])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[0], w1_area.m128_f32[0], w0_area.m128_f32[0]);
                __m128 u = _mm_mul_ps(texture_u, weights);
                u = _mm_mul_ps(u, _mm_set1_ps(depth.m128_f32[0]));
                u = _mm_mul_ps(u, _mm_set1_ps((float)render->tex_w));

                __m128 v = _mm_mul_ps(texture_v, weights);
                v = _mm_mul_ps(v, _mm_set1_ps(depth.m128_f32[0]));
                v = _mm_mul_ps(v, _mm_set1_ps((float)render->tex_h));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->bpp;
                Draw_Pixel_Pixel_Data(render->fmt, render->pixels, x + 3, y, pixelOffset);
            }

            depth = _mm_blendv_ps(previousDepthValue, depth, _mm_castsi128_ps(finalMask));
            depth = _mm_shuffle_ps(depth, depth, _MM_SHUFFLE(0, 1, 2, 3)); // reverse finalMask
            _mm_store_ps(&render->z_buffer_array[z_index], depth);
        }

        // One row step
        w0_row = _mm_add_epi32(w0_row, _mm_cvtps_epi32(B12));
        w1_row = _mm_add_epi32(w1_row, _mm_cvtps_epi32(B20));
        w2_row = _mm_add_epi32(w2_row, _mm_cvtps_epi32(B01));
    }
}