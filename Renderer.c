#include "Renderer.h"

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height)
{
    Renderer rend;

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
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
static inline void Draw_Pixel_RGBA(const Rendering_data *ren, int x, int y, uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
{
    // index = y * screen_w * x
    const int index = (int)y * ren->screen_width + (int)x;

    ren->pixels[index] = (Uint32)((alpha << 24) + (red << 16) + (green << 8) + (blue << 0));
    //    ren->pixels[index] = SDL_MapRGBA(ren->fmt,
    //                                     red,
    //                                     green,
    //                                     blue,
    //                                     alpha);
}

static void Draw_Pixel_Pixel_Data(const SDL_PixelFormat *fmt, unsigned int *pixels, int x, int y, const unsigned char *texture_data)
{
    // index = y * screen_w * x
    const int index = (int)y * 900 + (int)x;
    pixels[index] = SDL_MapRGBA(fmt,
                                (uint8_t)(texture_data[0]),
                                (uint8_t)(texture_data[1]),
                                (uint8_t)(texture_data[2]),
                                (uint8_t)(texture_data[3]));
}

static void Draw_Pixel_Pixel_Data_Light_Value(const SDL_PixelFormat *fmt, int screen_width, unsigned int *pixels, int x, int y, const unsigned char *texture_data, const float light_value)
{
    // index = y * screen_w * x
    const int index = (int)y * screen_width + (int)x;
    pixels[index] = SDL_MapRGBA(fmt,
                                (uint8_t)(texture_data[0] * light_value),
                                (uint8_t)(texture_data[1] * light_value),
                                (uint8_t)(texture_data[2] * light_value),
                                (uint8_t)(255));
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

void Draw_Depth_Buffer(const Rendering_data *render_data)
{
    const float z_max_depth = 100.0f;
    const __m128 max_depth = _mm_set1_ps(z_max_depth);
    const __m128 value_255 = _mm_set1_ps(255.0f);

    float *pDepthBuffer = render_data->z_buffer_array;

    int rowIdx = 0;
    for (unsigned int y = 0; y < render_data->screen_height; y += 2,
                      rowIdx += 2 * render_data->screen_width)
    {
        int index = rowIdx;
        for (unsigned int x = 0; x < render_data->screen_width; x += 2,
                          index += 4)
        {
            __m128 depthvalues = _mm_load_ps(&pDepthBuffer[index]);
            depthvalues = _mm_div_ps(depthvalues, max_depth);
            depthvalues = _mm_mul_ps(depthvalues, value_255);

            float shading[4];
            _mm_store_ps(shading, depthvalues);

            Draw_Pixel_RGBA(render_data, x + 0, y + 0, (uint8_t)shading[3], (uint8_t)shading[3], (uint8_t)shading[3], 255);
            Draw_Pixel_RGBA(render_data, x + 1, y + 0, (uint8_t)shading[2], (uint8_t)shading[2], (uint8_t)shading[2], 255);
            Draw_Pixel_RGBA(render_data, x + 0, y + 1, (uint8_t)shading[1], (uint8_t)shading[1], (uint8_t)shading[1], 255);
            Draw_Pixel_RGBA(render_data, x + 1, y + 1, (uint8_t)shading[0], (uint8_t)shading[0], (uint8_t)shading[0], 255);
            // Draw_Pixel_RGBA(&render_data, x + 0, y + 0, 255, 000, 000, 255);
            // Draw_Pixel_RGBA(&render_data, x + 1, y + 0, 255, 000, 000, 255);
            // Draw_Pixel_RGBA(&render_data, x + 0, y + 1, 255, 000, 000, 255);
            // Draw_Pixel_RGBA(&render_data, x + 1, y + 1, 255, 000, 000, 255);
        }
    }
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
    const __m128i max_values = _mm_cvtps_epi32(_mm_min_ps(_mm_max_ps(_mm_max_ps(v1, v2), v3), _mm_set_ps(0.0f, 0.0f, (float)screen_width - 1, (float)screen_height - 1)));
    const __m128i min_values = _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(_mm_min_ps(v1, v2), v3), _mm_set1_ps(0.0f)));

    // Returns {maxX, minX, maxY, minY}
    return _mm_unpacklo_epi32(max_values, min_values);
}

// https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
void Draw_Textured_Triangle(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2,
                            const __m128 texture_u, const __m128 texture_v,
                            const __m128 one_over_w1, const __m128 one_over_w2, const __m128 one_over_w3,
                            const __m128 frag_colour)
{
    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    // Skip triangle if area is zero
    if (triArea.m128i_i32[0] <= 0)
    {
        return;
    }

    // const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    const __m128i aa0Inc = _mm_slli_epi32(A0, 1);
    const __m128i aa1Inc = _mm_slli_epi32(A1, 1);
    const __m128i aa2Inc = _mm_slli_epi32(A2, 1);

    const __m128i bb0Inc = _mm_slli_epi32(B0, 1);
    const __m128i bb1Inc = _mm_slli_epi32(B1, 1);
    const __m128i bb2Inc = _mm_slli_epi32(B2, 1);

    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

    const __m128i col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

    __m128i row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    // Cast depth buffer to float
    float *pDepthBuffer = (float *)render->z_buffer_array;
    int rowIdx = (aabb.minY * render->screen_width + 2 * aabb.minX);

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += (2 * render->screen_width),
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Barycentric coordinates at start of row
        int index = rowIdx;
        __m128i alpha = sum0Row;
        __m128i beta = sum1Row;
        __m128i gama = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 index += 4,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 beta = _mm_add_epi32(beta, aa1Inc),
                 gama = _mm_add_epi32(gama, aa2Inc))
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
            const __m128i mask = _mm_or_si128(_mm_or_si128(alpha, beta), gama);
            const __m128i mask_check = _mm_cmplt_epi32(fxptZero, mask);

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask_check, mask_check))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(w0_area, one_over_w3);
            depth = _mm_add_ps(depth, _mm_mul_ps(w1_area, one_over_w2));
            depth = _mm_add_ps(depth, _mm_mul_ps(w2_area, one_over_w1));
            depth = _mm_rcp_ps(depth);

            //// DEPTH BUFFER
            const __m128 previousDepthValue = _mm_load_ps(&pDepthBuffer[index]);

            __m128 get_the_zero_value_index = _mm_cmpeq_ps(previousDepthValue, _mm_setzero_ps());
            __m128 fill_zero_values_with_depths = _mm_blendv_ps(previousDepthValue, depth, get_the_zero_value_index);
            __m128 which_depths_values_are_min = _mm_cmplt_ps(depth, fill_zero_values_with_depths); // ( a[i+31:i] < b[i+31:i] ) ? 0xFFFFFFFF : 0

            __m128 final_mask = _mm_or_ps(_mm_castsi128_ps(mask_check), which_depths_values_are_min);

            __m128 final_depth_values = _mm_blendv_ps(which_depths_values_are_min, previous_depth, _mm_castsi128_ps(mask_check));

            // Precalulate uv constants
            const __m128 depth_w = _mm_mul_ps(depth, _mm_set1_ps((float)render->tex_w - 1));
            const __m128 depth_h = _mm_mul_ps(depth, _mm_set1_ps((float)render->tex_h - 1));

            if (which_depth_to_draw.m128i_i32[3])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[3], w1_area.m128_f32[3], w0_area.m128_f32[3]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(3, 3, 3, 3)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(3, 3, 3, 3)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;

                const uint8_t red = (uint8_t)(pixelOffset[0]);
                const uint8_t gre = (uint8_t)(pixelOffset[1]);
                const uint8_t blu = (uint8_t)(pixelOffset[2]);
                const uint8_t alp = (uint8_t)(render->tex_bpp == 4 ? pixelOffset[3] : 255);

                Draw_Pixel_RGBA(render, x + 0, y + 0, red, gre, blu, alp);
            }

            if (which_depth_to_draw.m128i_i32[2])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[2], w1_area.m128_f32[2], w0_area.m128_f32[2]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(2, 2, 2, 2)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(2, 2, 2, 2)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;

                const uint8_t red = (uint8_t)(pixelOffset[0]);
                const uint8_t gre = (uint8_t)(pixelOffset[1]);
                const uint8_t blu = (uint8_t)(pixelOffset[2]);
                const uint8_t alp = (uint8_t)(render->tex_bpp == 4 ? pixelOffset[3] : 255);

                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
            }

            if (which_depth_to_draw.m128i_i32[1])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[1], w1_area.m128_f32[1], w0_area.m128_f32[1]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(1, 1, 1, 1)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(1, 1, 1, 1)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;

                const uint8_t red = (uint8_t)(pixelOffset[0]);
                const uint8_t gre = (uint8_t)(pixelOffset[1]);
                const uint8_t blu = (uint8_t)(pixelOffset[2]);
                const uint8_t alp = (uint8_t)(render->tex_bpp == 4 ? pixelOffset[3] : 255);

                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
            }

            if (which_depth_to_draw.m128i_i32[0])
            {
                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[0], w1_area.m128_f32[0], w0_area.m128_f32[0]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(0, 0, 0, 0)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(0, 0, 0, 0)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const unsigned char *pixelOffset = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;

                const uint8_t red = (uint8_t)(pixelOffset[0]);
                const uint8_t gre = (uint8_t)(pixelOffset[1]);
                const uint8_t blu = (uint8_t)(pixelOffset[2]);
                const uint8_t alp = (uint8_t)(render->tex_bpp == 4 ? pixelOffset[3] : 255);

                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }
        }
    }
}

void Draw_Textured_Shaded_Triangle(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2, const __m128 frag_colour)
{
    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    __m128 v0_z = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 v1_z = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 v2_z = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 3, 3, 3));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    const __m128i p_x = _mm_add_epi32(_mm_set1_epi32(aabb.minX), _mm_set_epi32(0, 1, 2, 3));
    const __m128i p_y = _mm_set1_epi32(aabb.minY);

    // Barycentric Setip
    // Order of triangle sides *IMPORTANT*
    // v1, v2 :  w0_row = (A12 * p.x) + (B12 * p.y) + C12;
    // v2, v0 :  w1_row = (A20 * p.x) + (B20 * p.y) + C20;
    // v0, v1 :  w2_row = (A01 * p.x) + (B01 * p.y) + C01;
    __m128i w0_row = _mm_add_epi32(
        _mm_add_epi32(
            _mm_mullo_epi32(A0, p_x), _mm_mullo_epi32(B0, p_y)),
        C0);
    __m128i w1_row = _mm_add_epi32(
        _mm_add_epi32(
            _mm_mullo_epi32(A1, p_x), _mm_mullo_epi32(B1, p_y)),
        C1);
    __m128i w2_row = _mm_add_epi32(
        _mm_add_epi32(
            _mm_mullo_epi32(A2, p_x), _mm_mullo_epi32(B2, p_y)),
        C2);

    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    // X Step
    const __m128i X_Step_w0 = _mm_mullo_epi32(A0, _mm_set1_epi32(4));
    const __m128i X_Step_w1 = _mm_mullo_epi32(A1, _mm_set1_epi32(4));
    const __m128i X_Step_w2 = _mm_mullo_epi32(A2, _mm_set1_epi32(4));

    float *pDepthBuffer = (float *)render->z_buffer_array;

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 1,
             w0_row = _mm_add_epi32(w0_row, B0),
             w1_row = _mm_add_epi32(w1_row, B1),
             w2_row = _mm_add_epi32(w2_row, B2))
    {
        // Barycentric coordinates at start of row
        __m128i w0 = w0_row;
        __m128i w1 = w1_row;
        __m128i w2 = w2_row;

        for (int x = aabb.minX; x < aabb.maxX; x += 4,
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
            __m128 depth = _mm_mul_ps(w0_area, v0_z);
            depth = _mm_add_ps(depth, _mm_mul_ps(w1_area, v1_z));
            depth = _mm_add_ps(depth, _mm_mul_ps(w2_area, v2_z));
            depth = _mm_rcp_ps(depth);

            const int z_index = x + render->screen_width * y;

            //__m128 previousDepthValue = _mm_load_ps(&pDepthBuffer[z_index]);
            // previousDepthValue = _mm_shuffle_ps(previousDepthValue, previousDepthValue, _MM_SHUFFLE(0, 1, 2, 3));
            // const __m128 depthMask = _mm_cmpge_ps(depth, previousDepthValue); // dst[i+31:i] := ( a[i+31:i] >= b[i+31:i] ) ? 0xFFFFFFFF : 0

            // mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(0, 1, 2, 3)); // reverse the mask
            // const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(depthMask));
            const __m128i finalMask = mask;

            float light_thing[4];
            _mm_store_ps(light_thing, frag_colour);

            const __m128 colour = frag_colour;

            if (finalMask.m128i_i32[3])
            {
                // const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[3], w1_area.m128_f32[3], w0_area.m128_f32[3]);
                // const __m128 colour = _mm_mul_ps(weights, frag_colour);

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[2])
            {
                // const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[3], w1_area.m128_f32[3], w0_area.m128_f32[3]);
                // const __m128 colour = _mm_mul_ps(weights, frag_colour);

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[1])
            {
                // const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[1], w1_area.m128_f32[1], w0_area.m128_f32[1]);
                // const __m128 colour = _mm_mul_ps(weights, frag_colour);

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 2, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[0])
            // if (mask.m128i_i32[0] && finalMask.m128i_i32[0])
            {
                // const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[0], w1_area.m128_f32[0], w0_area.m128_f32[0]);
                // const __m128 colour = _mm_mul_ps(weights, frag_colour);

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 3, y, red, gre, blu, alp);
            }

            // depth = _mm_blendv_ps(previousDepthValue, depth, _mm_castsi128_ps(finalMask));
            // depth = _mm_shuffle_ps(depth, depth, _MM_SHUFFLE(0, 1, 2, 3)); // reverse finalMask
            //_mm_store_ps(&pDepthBuffer[z_index], depth);
        }
    }
}

void Draw_Specular_Shaded(const Rendering_data *render, const __m128 *screen_position_vertixies, const __m128 *world_position_verticies,
                          const __m128 *normal_vectors, const PointLight pl)
{
    const __m128 v0 = screen_position_vertixies[2];
    const __m128 v1 = screen_position_vertixies[1];
    const __m128 v2 = screen_position_vertixies[0];

    __m128 wp0 = world_position_verticies[2];
    __m128 wp1 = world_position_verticies[1];
    __m128 wp2 = world_position_verticies[0];

    __m128 nrm0 = normal_vectors[2];
    __m128 nrm1 = normal_vectors[1];
    __m128 nrm2 = normal_vectors[0];

    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    // Compute triangle area
    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    // const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    const int lane = 0;

    __m128i aa0 = _mm_set1_epi32(A0.m128i_i32[lane]);
    __m128i aa1 = _mm_set1_epi32(A1.m128i_i32[lane]);
    __m128i aa2 = _mm_set1_epi32(A2.m128i_i32[lane]);

    __m128i bb0 = _mm_set1_epi32(B0.m128i_i32[lane]);
    __m128i bb1 = _mm_set1_epi32(B1.m128i_i32[lane]);
    __m128i bb2 = _mm_set1_epi32(B2.m128i_i32[lane]);

    __m128i aa0Inc = _mm_slli_epi32(aa0, 1);
    __m128i aa1Inc = _mm_slli_epi32(aa1, 1);
    __m128i aa2Inc = _mm_slli_epi32(aa2, 1);

    __m128i row, col;
    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

    col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    __m128i aa0Col = _mm_mullo_epi32(aa0, col);
    __m128i aa1Col = _mm_mullo_epi32(aa1, col);
    __m128i aa2Col = _mm_mullo_epi32(aa2, col);

    row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(bb0, row), _mm_set1_epi32(C0.m128i_i32[lane]));
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(bb1, row), _mm_set1_epi32(C1.m128i_i32[lane]));
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(bb2, row), _mm_set1_epi32(C2.m128i_i32[lane]));

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    __m128i bb0Inc = _mm_slli_epi32(bb0, 1);
    __m128i bb1Inc = _mm_slli_epi32(bb1, 1);
    __m128i bb2Inc = _mm_slli_epi32(bb2, 1);

    float *pDepthBuffer = (float *)render->z_buffer_array;

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Compute barycentric coordinates
        __m128i alpha = sum0Row;
        __m128i betaa = sum1Row;
        __m128i gamaa = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 betaa = _mm_add_epi32(betaa, aa1Inc),
                 gamaa = _mm_add_epi32(gamaa, aa2Inc))
        // One step to the right
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, betaa), gamaa));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(betaa), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gamaa), oneOverTriArea);
            // const __m128 w0_area = _mm_cvtepi32_ps(alpha);
            // const __m128 w1_area = _mm_cvtepi32_ps(betaa);
            // const __m128 w2_area = _mm_cvtepi32_ps(gamaa);

            // const int z_index = x + render->screen_width * y;

            //__m128 previousDepthValue = _mm_load_ps(&pDepthBuffer[z_index]);
            // previousDepthValue = _mm_shuffle_ps(previousDepthValue, previousDepthValue, _MM_SHUFFLE(0, 1, 2, 3));
            // const __m128 depthMask = _mm_cmpge_ps(depth, previousDepthValue); // dst[i+31:i] := ( a[i+31:i] >= b[i+31:i] ) ? 0xFFFFFFFF : 0

            // mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(0, 1, 2, 3)); // reverse the mask
            // const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(depthMask));
            const __m128i finalMask = mask;

            if (finalMask.m128i_i32[3])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[3]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[3]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[3]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, nrm0),
                        _mm_mul_ps(weight2, nrm1)),
                    _mm_mul_ps(weight3, nrm2));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y + 0, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[2])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[2]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[2]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[2]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, nrm0),
                        _mm_mul_ps(weight2, nrm1)),
                    _mm_mul_ps(weight3, nrm2));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[1])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, nrm0),
                        _mm_mul_ps(weight2, nrm1)),
                    _mm_mul_ps(weight3, nrm2));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[0])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[0]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[0]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[0]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, nrm0),
                        _mm_mul_ps(weight2, nrm1)),
                    _mm_mul_ps(weight3, nrm2));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }

            // depth = _mm_blendv_ps(previousDepthValue, depth, _mm_castsi128_ps(finalMask));
            // depth = _mm_shuffle_ps(depth, depth, _MM_SHUFFLE(0, 1, 2, 3)); // reverse finalMask
            //_mm_store_ps(&pDepthBuffer[z_index], depth);
        }
    }
}

void Draw_Textured_Smooth_Shaded_Triangle(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2, __m128 nrm1, __m128 nrm2, __m128 nrm3, const __m128 frag_colour, const __m128 light_direction)
{
    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    // This is really the 1/w value?
    __m128 v0_z = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1_z = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v2_z = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    // const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    v0_z = _mm_mul_ps(oneOverTriArea, v0_z); // z[0] *= oneOverTotalArea;
    v1_z = _mm_mul_ps(oneOverTriArea, v1_z); // z[1] *= oneOverTotalArea;
    v2_z = _mm_mul_ps(oneOverTriArea, v2_z); // z[2] *= oneOverTotalArea;

    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

    const __m128i aa0Inc = _mm_slli_epi32(A0, 1);
    const __m128i aa1Inc = _mm_slli_epi32(A1, 1);
    const __m128i aa2Inc = _mm_slli_epi32(A2, 1);

    const __m128i col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

    const __m128i bb0Inc = _mm_slli_epi32(B0, 1);
    const __m128i bb1Inc = _mm_slli_epi32(B1, 1);
    const __m128i bb2Inc = _mm_slli_epi32(B2, 1);

    __m128i row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    // Tranverse pixels in 2x2 blocks and store 2x2 pixel quad depths contiguously in memory ==> 2*X
    // This method provides better perfromance
    int rowIdx = (aabb.minY * render->screen_width + 2 * aabb.minX);

    // Cast depth buffer to float
    float *pDepthBuffer = (float *)render->z_buffer_array;

    // Lights
    const __m128 light_colour = _mm_set1_ps(1.0f);

    float dp_nrm1 = Calculate_Dot_Product_SIMD(nrm1, light_direction);
    float dp_nrm2 = Calculate_Dot_Product_SIMD(nrm2, light_direction);
    float dp_nrm3 = Calculate_Dot_Product_SIMD(nrm3, light_direction);

    dp_nrm1 = (float)fmax((double)dp_nrm1, 0.0);
    dp_nrm2 = (float)fmax((double)dp_nrm2, 0.0);
    dp_nrm3 = (float)fmax((double)dp_nrm3, 0.0);

    const __m128 ambient = _mm_set1_ps(0.3f);

    const __m128 diffuse1 = _mm_mul_ps(light_colour, _mm_set1_ps(dp_nrm1));
    const __m128 diffuse2 = _mm_mul_ps(light_colour, _mm_set1_ps(dp_nrm2));
    const __m128 diffuse3 = _mm_mul_ps(light_colour, _mm_set1_ps(dp_nrm3));

    __m128 colour1 = Clamp_m128(_mm_add_ps(ambient, diffuse1), 0.0f, 1.0f);
    __m128 colour2 = Clamp_m128(_mm_add_ps(ambient, diffuse2), 0.0f, 1.0f);
    __m128 colour3 = Clamp_m128(_mm_add_ps(ambient, diffuse3), 0.0f, 1.0f);

    colour1 = _mm_mul_ps(colour1, _mm_set1_ps(255.0f));
    colour2 = _mm_mul_ps(colour2, _mm_set1_ps(255.0f));
    colour3 = _mm_mul_ps(colour3, _mm_set1_ps(255.0f));

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += (2 * render->screen_width),
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Barycentric coordinates at start of row
        int index = rowIdx;
        __m128i alpha = sum0Row;
        __m128i beta = sum1Row;
        __m128i gama = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 index += 4,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 beta = _mm_add_epi32(beta, aa1Inc),
                 gama = _mm_add_epi32(gama, aa2Inc))
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, beta), gama));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(_mm_cvtepi32_ps(alpha), v0_z);
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(beta), v1_z));
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(gama), v2_z));
            depth = _mm_rcp_ps(depth);

            const __m128 previousDepthValue = _mm_load_ps(&pDepthBuffer[index]);
            const __m128 mergedDepth = _mm_max_ps(depth, previousDepthValue);
            const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(mergedDepth));

            if (finalMask.m128i_i32[3])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[3]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[3]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[3]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[2])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[2]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[2]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[2]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[1])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[0])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }

            const __m128 finaldepth = _mm_blendv_ps(mergedDepth, previousDepthValue, _mm_cvtepi32_ps(finalMask));
            _mm_store_ps(&pDepthBuffer[index], finaldepth);
        }
    }
}

void Draw_Triangle_With_Colour(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2,
                               const __m128 colour1, const __m128 colour2, const __m128 colour3)
{
    // used when checking if w0,w1,w2 is greater than 0;
    __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    __m128 v0_z = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1_z = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v2_z = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    // const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    v0_z = _mm_mul_ps(oneOverTriArea, v0_z); // z[0] *= oneOverTotalArea;
    v1_z = _mm_mul_ps(oneOverTriArea, v1_z); // z[1] *= oneOverTotalArea;
    v2_z = _mm_mul_ps(oneOverTriArea, v2_z); // z[2] *= oneOverTotalArea;

    const __m128i aa0Inc = _mm_slli_epi32(A0, 1);
    const __m128i aa1Inc = _mm_slli_epi32(A1, 1);
    const __m128i aa2Inc = _mm_slli_epi32(A2, 1);

    const __m128i bb0Inc = _mm_slli_epi32(B0, 1);
    const __m128i bb1Inc = _mm_slli_epi32(B1, 1);
    const __m128i bb2Inc = _mm_slli_epi32(B2, 1);

    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

    const __m128i col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

    __m128i row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    // Tranverse pixels in 2x2 blocks and store 2x2 pixel quad depths contiguously in memory ==> 2*X
    // This method provides better perfromance
    int rowIdx = (aabb.minY * render->screen_width + 2 * aabb.minX);

    // Cast depth buffer to float
    float *pDepthBuffer = (float *)render->z_buffer_array;

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += (2 * render->screen_width),
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Barycentric coordinates at start of row
        int index = rowIdx;
        __m128i alpha = sum0Row;
        __m128i beta = sum1Row;
        __m128i gama = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 index += 4,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 beta = _mm_add_epi32(beta, aa1Inc),
                 gama = _mm_add_epi32(gama, aa2Inc))
        {
            // Stepping through in a 2x2 square of pixels

            // Test Pixel inside triangle
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, beta), gama));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(_mm_cvtepi32_ps(alpha), v0_z);
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(beta), v1_z));
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(gama), v2_z));
            // depth = _mm_rcp_ps(depth);

            const __m128 previousDepthValue = _mm_load_ps(&pDepthBuffer[index]);
            const __m128 mergedDepth = _mm_max_ps(depth, previousDepthValue);
            const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(mergedDepth));

            if (finalMask.m128i_i32[3])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[3]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[3]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[3]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[2])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[2]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[2]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[2]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[1])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[0])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[0]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[0]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[0]);

                const __m128 colour = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, colour1),
                        _mm_mul_ps(weight2, colour2)),
                    _mm_mul_ps(weight3, colour3));

                const uint8_t red = (uint8_t)(colour.m128_f32[0]);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1]);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2]);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }

            const __m128 finaldepth = _mm_blendv_ps(mergedDepth, previousDepthValue, _mm_cvtepi32_ps(finalMask));
            _mm_store_ps(&pDepthBuffer[index], finaldepth);
        }
    }
}

void Draw_Triangle_Per_Pixel(const Rendering_data *render, const __m128 *screen_position_vertixies, const __m128 *world_position_verticies,
                             const __m128 *normal_vectors, const PointLight pl)
{
    __m128 v0 = screen_position_vertixies[2];
    __m128 v1 = screen_position_vertixies[1];
    __m128 v2 = screen_position_vertixies[0];

    __m128 normal1 = normal_vectors[0];
    __m128 normal2 = normal_vectors[1];
    __m128 normal3 = normal_vectors[2];

    __m128 world_pos1 = world_position_verticies[0];
    __m128 world_pos2 = world_position_verticies[1];
    __m128 world_pos3 = world_position_verticies[2];

    // used when checking if w0,w1,w2 is greater than 0;
    __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    __m128 v0_z = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1_z = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v2_z = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    // const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    v0_z = _mm_mul_ps(oneOverTriArea, v0_z); // z[0] *= oneOverTotalArea;
    v1_z = _mm_mul_ps(oneOverTriArea, v1_z); // z[1] *= oneOverTotalArea;
    v2_z = _mm_mul_ps(oneOverTriArea, v2_z); // z[2] *= oneOverTotalArea;

    const __m128i aa0Inc = _mm_slli_epi32(A0, 1);
    const __m128i aa1Inc = _mm_slli_epi32(A1, 1);
    const __m128i aa2Inc = _mm_slli_epi32(A2, 1);

    const __m128i bb0Inc = _mm_slli_epi32(B0, 1);
    const __m128i bb1Inc = _mm_slli_epi32(B1, 1);
    const __m128i bb2Inc = _mm_slli_epi32(B2, 1);

    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

    const __m128i col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

    __m128i row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    // Tranverse pixels in 2x2 blocks and store 2x2 pixel quad depths contiguously in memory ==> 2*X
    // This method provides better perfromance
    int rowIdx = (aabb.minY * render->screen_width + 2 * aabb.minX);

    // Cast depth buffer to float
    float *pDepthBuffer = (float *)render->z_buffer_array;

    const __m128 ambient = _mm_set1_ps(0.1f);

    // const __m128 camera_position = _mm_setzero_ps();

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += (2 * render->screen_width),
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Barycentric coordinates at start of row
        int index = rowIdx;
        __m128i alpha = sum0Row;
        __m128i beta = sum1Row;
        __m128i gama = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 index += 4,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 beta = _mm_add_epi32(beta, aa1Inc),
                 gama = _mm_add_epi32(gama, aa2Inc))
        {
            // Stepping through in a 2x2 square of pixels

            // Test Pixel inside triangle
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, beta), gama));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(_mm_cvtepi32_ps(alpha), v0_z);
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(beta), v1_z));
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(gama), v2_z));
            // depth = _mm_rcp_ps(depth);

            const __m128 previousDepthValue = _mm_load_ps(&pDepthBuffer[index]);
            const __m128 mergedDepth = _mm_max_ps(depth, previousDepthValue);
            const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(mergedDepth));

            if (finalMask.m128i_i32[3])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[3]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[3]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[3]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, normal1),
                        _mm_mul_ps(weight2, normal2)),
                    _mm_mul_ps(weight3, normal3));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, world_pos1),
                        _mm_mul_ps(weight2, world_pos2)),
                    _mm_mul_ps(weight3, world_pos3));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = _mm_add_ps(colour, ambient);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[2])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[2]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[2]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[2]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, normal1),
                        _mm_mul_ps(weight2, normal2)),
                    _mm_mul_ps(weight3, normal3));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, world_pos1),
                        _mm_mul_ps(weight2, world_pos2)),
                    _mm_mul_ps(weight3, world_pos3));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = _mm_add_ps(colour, ambient);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[1])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, normal1),
                        _mm_mul_ps(weight2, normal2)),
                    _mm_mul_ps(weight3, normal3));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, world_pos1),
                        _mm_mul_ps(weight2, world_pos2)),
                    _mm_mul_ps(weight3, world_pos3));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = _mm_add_ps(colour, ambient);
                colour = Clamp_m128(colour, 0.0f, 1.0f);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[0])
            {
                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[0]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[0]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[0]);

                const __m128 normal = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, normal1),
                        _mm_mul_ps(weight2, normal2)),
                    _mm_mul_ps(weight3, normal3));

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, world_pos1),
                        _mm_mul_ps(weight2, world_pos2)),
                    _mm_mul_ps(weight3, world_pos3));

                __m128 colour = Calculate_Point_Light_Colour(pl, normal, position);
                colour = _mm_add_ps(colour, ambient);
                colour = Clamp_m128(colour, 0.0f, 1.0f);
                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }

            const __m128 finaldepth = _mm_blendv_ps(mergedDepth, previousDepthValue, _mm_cvtepi32_ps(finalMask));
            _mm_store_ps(&pDepthBuffer[index], finaldepth);
        }
    }
}

void Draw_Normal_Mapped_Triangle(const Rendering_data *render, const __m128 *screen_position_vertixies, __m128 *world_position_verticies,
                                 const __m128 texture_u, const __m128 texture_v,
                                 const __m128 one_over_w1, const __m128 one_over_w2, const __m128 one_over_w3,
                                 const Mat4x4 TBN)
{
    __m128 v0 = screen_position_vertixies[2];
    __m128 v1 = screen_position_vertixies[1];
    __m128 v2 = screen_position_vertixies[0];

    __m128 wp0 = world_position_verticies[2];
    __m128 wp1 = world_position_verticies[1];
    __m128 wp2 = world_position_verticies[0];

    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    __m128 v0_z = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1_z = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v2_z = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

    __m128i triArea = _mm_mullo_epi32(B2, A1);
    triArea = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

    // const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    v0_z = _mm_mul_ps(oneOverTriArea, v0_z); // z[0] *= oneOverTotalArea;
    v1_z = _mm_mul_ps(oneOverTriArea, v1_z); // z[1] *= oneOverTotalArea;
    v2_z = _mm_mul_ps(oneOverTriArea, v2_z); // z[2] *= oneOverTotalArea;

    const int lane = 0;

    __m128i aa0 = _mm_set1_epi32(A0.m128i_i32[lane]);
    __m128i aa1 = _mm_set1_epi32(A1.m128i_i32[lane]);
    __m128i aa2 = _mm_set1_epi32(A2.m128i_i32[lane]);

    __m128i bb0 = _mm_set1_epi32(B0.m128i_i32[lane]);
    __m128i bb1 = _mm_set1_epi32(B1.m128i_i32[lane]);
    __m128i bb2 = _mm_set1_epi32(B2.m128i_i32[lane]);

    __m128i aa0Inc = _mm_slli_epi32(aa0, 1);
    __m128i aa1Inc = _mm_slli_epi32(aa1, 1);
    __m128i aa2Inc = _mm_slli_epi32(aa2, 1);

    __m128i row, col;
    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

    col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    __m128i aa0Col = _mm_mullo_epi32(aa0, col);
    __m128i aa1Col = _mm_mullo_epi32(aa1, col);
    __m128i aa2Col = _mm_mullo_epi32(aa2, col);

    row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(bb0, row), _mm_set1_epi32(C0.m128i_i32[lane]));
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(bb1, row), _mm_set1_epi32(C1.m128i_i32[lane]));
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(bb2, row), _mm_set1_epi32(C2.m128i_i32[lane]));

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    __m128i bb0Inc = _mm_slli_epi32(bb0, 1);
    __m128i bb1Inc = _mm_slli_epi32(bb1, 1);
    __m128i bb2Inc = _mm_slli_epi32(bb2, 1);

    // Taangent Space Calculations
    const __m128 light_position = _mm_set_ps(0.0f, 2.0f, 0.0f, -1.0f);
    const __m128 view_position = _mm_set1_ps(0.0f); // for specular calculation

    const __m128 Tangent_Light_Pos = Matrix_Multiply_Vector_SIMD(TBN.elements, light_position);
    const __m128 Tangent_View_Pos = Matrix_Multiply_Vector_SIMD(TBN.elements, view_position); // for specular

    wp0 = Matrix_Multiply_Vector_SIMD(TBN.elements, wp0);
    wp1 = Matrix_Multiply_Vector_SIMD(TBN.elements, wp1);
    wp2 = Matrix_Multiply_Vector_SIMD(TBN.elements, wp2);

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Compute barycentric coordinates
        __m128i alpha = sum0Row;
        __m128i betaa = sum1Row;
        __m128i gamaa = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 betaa = _mm_add_epi32(betaa, aa1Inc),
                 gamaa = _mm_add_epi32(gamaa, aa2Inc))
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, betaa), gamaa));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(betaa), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gamaa), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(_mm_cvtepi32_ps(alpha), one_over_w3);
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(betaa), one_over_w2));
            depth = _mm_add_ps(depth, _mm_mul_ps(_mm_cvtepi32_ps(gamaa), one_over_w1));
            depth = _mm_rcp_ps(depth);

            // const int z_index = x + render->screen_width * y;

            //__m128 previousDepthValue = _mm_load_ps(&render->z_buffer_array[z_index]);
            // previousDepthValue = _mm_shuffle_ps(previousDepthValue, previousDepthValue, _MM_SHUFFLE(0, 1, 2, 3));
            // const __m128 depthMask = _mm_cmpge_ps(depth, previousDepthValue); // dst[i+31:i] := ( a[i+31:i] >= b[i+31:i] ) ? 0xFFFFFFFF : 0
            // early out depth mask check
            // if (_mm_test_all_zeros(depthMask, depthMask))
            //     continue;

            //  mask = _mm_shuffle_epi32(mask, _MM_SHUFFLE(0, 1, 2, 3)); // reverse the mask
            //  const __m128i finalMask = _mm_and_si128(mask, _mm_castps_si128(depthMask));
            const __m128i finalMask = mask;

            const __m128 depth_w = _mm_mul_ps(depth, _mm_set1_ps((float)render->tex_w - 1));
            const __m128 depth_h = _mm_mul_ps(depth, _mm_set1_ps((float)render->tex_h - 1));

            if (finalMask.m128i_i32[3])
            {
                const __m128 weights = _mm_set_ps(0.0f, (float)gamaa.m128i_i32[3], (float)betaa.m128i_i32[3], (float)alpha.m128i_i32[3]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(3, 3, 3, 3)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(3, 3, 3, 3)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[3]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[3]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[3]);

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                const unsigned char *diffuse_texture = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
                const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;

                const __m128 colour = Calculate_Normal_Mapping_Colour(diffuse_texture, normal_texture, TBN, position, Tangent_Light_Pos, Tangent_View_Pos);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[2])
            {
                const __m128 weights = _mm_set_ps(0.0f, (float)gamaa.m128i_i32[2], (float)betaa.m128i_i32[2], (float)alpha.m128i_i32[2]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(2, 2, 2, 2)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(2, 2, 2, 2)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[2]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[2]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[2]);

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                const unsigned char *diffuse_texture = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
                const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;

                const __m128 colour = Calculate_Normal_Mapping_Colour(diffuse_texture, normal_texture, TBN, position, Tangent_Light_Pos, Tangent_View_Pos);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[1])
            {
                const __m128 weights = _mm_set_ps(0.0f, (float)gamaa.m128i_i32[1], (float)betaa.m128i_i32[1], (float)alpha.m128i_i32[1]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(1, 1, 1, 1)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(1, 1, 1, 1)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                const unsigned char *diffuse_texture = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
                const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;

                const __m128 colour = Calculate_Normal_Mapping_Colour(diffuse_texture, normal_texture, TBN, position, Tangent_Light_Pos, Tangent_View_Pos);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
            }

            if (finalMask.m128i_i32[0])
            {
                const __m128 weights = _mm_set_ps(0.0f, (float)gamaa.m128i_i32[0], (float)betaa.m128i_i32[0], (float)alpha.m128i_i32[0]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(0, 0, 0, 0)));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(0, 0, 0, 0)));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[0]);
                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[0]);
                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[0]);

                const __m128 position = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(weight1, wp0),
                        _mm_mul_ps(weight2, wp1)),
                    _mm_mul_ps(weight3, wp2));

                const unsigned char *diffuse_texture = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
                const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;

                const __m128 colour = Calculate_Normal_Mapping_Colour(diffuse_texture, normal_texture, TBN, position, Tangent_Light_Pos, Tangent_View_Pos);

                const uint8_t red = (uint8_t)(colour.m128_f32[0] * 255);
                const uint8_t gre = (uint8_t)(colour.m128_f32[1] * 255);
                const uint8_t blu = (uint8_t)(colour.m128_f32[2] * 255);
                const uint8_t alp = (uint8_t)(255);

                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }

            // depth = _mm_blendv_ps(previousDepthValue, depth, _mm_castsi128_ps(finalMask));
            // depth = _mm_shuffle_ps(depth, depth, _MM_SHUFFLE(0, 1, 2, 3)); // reverse finalMask
            //_mm_store_ps(&render->z_buffer_array[z_index], depth);
        }
    }
}