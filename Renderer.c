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

// https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
static __m128i Edge_init(const __m128 v0, const __m128 v1, unsigned int minX, unsigned int minY, __m128i *return_oneStepX)
{
    // int A = v0.y - v1.y;
    // int B = v1.x - v0.x;
    // int C = v0.x * v1.y - v0.y * v1.x;
    const __m128 A = _mm_sub_ps(v0, v1);
    const __m128 B = _mm_sub_ps(v1, v0);

    const __m128 AB = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 1, 0, 1));
    //__declspec(align(16)) float tmp[4];
    float tmp[4];
    _mm_store_ps(tmp, AB);

    const float C = tmp[1] * tmp[0] - tmp[1] * tmp[0];

    // Step deltas
    // oneStepX = Vec4i(A * stepXSize);
    // oneStepY = Vec4i(B * stepYSize);

    // x/y values for initial pixel block
    // Vec4i x = Vec4i(origin.x) + Vec4i(0, 1, 2, 3);
    // Vec4i y = Vec4i(origin.y);

    // const int stepXSize = 4;
    // const int stepYSize = 1;
    const __m128i x = _mm_add_epi32(_mm_set1_epi32(minX), _mm_set_epi32(0, 1, 2, 3));
    const __m128i y = _mm_set1_epi32(minY);

    *return_oneStepX = x;

    // Edge function values at origin
    // return Vec4i(A) * x + Vec4i(B) * y + Vec4i(C);

    const __m128i res = _mm_add_epi32(
        _mm_add_epi32(
            _mm_mul_epi32(_mm_castps_si128(_mm_shuffle_ps(A, A, _MM_SHUFFLE(0, 0, 0, 0))), x),
            _mm_mul_epi32(_mm_castps_si128(_mm_shuffle_ps(B, B, _MM_SHUFFLE(0, 0, 0, 0))), y)),
        _mm_set1_epi32((int)C));

    return res;
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

static __m128i Get_AABB_SIMD(const __m128 v1, const __m128 v2, const __m128 v3)
{
    const __m128i max_values = _mm_cvtps_epi32(_mm_min_ps(_mm_max_ps(_mm_max_ps(v1, v2), v3), _mm_set_ps(0.0f, 0.0f, 512.0f, 512.0f)));
    const __m128i min_values = _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(_mm_min_ps(v1, v2), v3), _mm_set1_ps(0.0f)));

    // Returns {maxX, minX, maxY, minY}
    return _mm_unpacklo_epi32(max_values, min_values);
}

void Draw_Function(const Rendering_data *render, const __m128 v0, const __m128 v1, const __m128 v2)
{
    // Random colours for each point
    // const __m128i colour_red_value = _mm_set_epi32(rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1);
    // const __m128i colour_green_value = _mm_set_epi32(rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1);
    // const __m128i colour_blue_value = _mm_set_epi32(rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1);

    __m128 texture1 = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    __m128 texture2 = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    __m128 texture3 = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);

    // used when checking if w0,w1,w2 is greater than 0;
    __m128i fxptZero = _mm_setzero_si128();

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2));

    // X and Y value setup
    const __m128 v0_x = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 v1_x = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 v2_x = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0));

    const __m128 v0_y = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 v1_y = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 v2_y = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1));

    __m128 v0_z = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1_z = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v2_z = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));

    v0_z = _mm_div_ps(_mm_set1_ps(1.0f), v0_z);
    v1_z = _mm_div_ps(_mm_set1_ps(1.0f), v1_z);
    v2_z = _mm_div_ps(_mm_set1_ps(1.0f), v2_z);

    texture1 = _mm_mul_ps(texture1, v0_z);
    texture2 = _mm_mul_ps(texture2, v1_z);
    texture3 = _mm_mul_ps(texture3, v2_z);

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
    __m128 w0_row = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(A12, p_x), _mm_mul_ps(B12, p_y)),
        C12);
    __m128 w1_row = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(A20, p_x), _mm_mul_ps(B20, p_y)),
        C20);
    __m128 w2_row = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(A01, p_x), _mm_mul_ps(B01, p_y)),
        C01);

    // Compute triangle area
    __m128 triArea = _mm_mul_ps(A01, v2_x);
    triArea = _mm_add_ps(triArea, _mm_mul_ps(B01, v2_y));
    triArea = _mm_add_ps(triArea, C01);

    const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), triArea);

    // X Step
    const __m128 X_Step_w0 = _mm_mul_ps(A12, _mm_set1_ps(4.0f));
    const __m128 X_Step_w1 = _mm_mul_ps(A20, _mm_set1_ps(4.0f));
    const __m128 X_Step_w2 = _mm_mul_ps(A01, _mm_set1_ps(4.0f));
    // Y Step
    // const __m128i Y_Step = 1;

    // Precompute the Z
    v0_z = _mm_mul_ps(v0_z, oneOverTriArea);
    v1_z = _mm_mul_ps(v1_z, oneOverTriArea);
    v2_z = _mm_mul_ps(v2_z, oneOverTriArea);

    // Rasterize
    for (int y = aabb.minY; y <= aabb.maxY; y += 1)
    {
        // Barycentric coordinates at start of row
        __m128 w0 = w0_row;
        __m128 w1 = w1_row;
        __m128 w2 = w2_row;

        for (int x = aabb.minX; x <= aabb.maxX; x += 4,
                 w0 = _mm_add_ps(w0, X_Step_w0),
                 w1 = _mm_add_ps(w1, X_Step_w1),
                 w2 = _mm_add_ps(w2, X_Step_w2))
        // One step to the right
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(_mm_cvtps_epi32(w0), _mm_cvtps_epi32(w1)), _mm_cvtps_epi32(w2)));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(w0, v0_z);
            depth = _mm_add_ps(depth, _mm_mul_ps(w1, v1_z));
            depth = _mm_add_ps(depth, _mm_mul_ps(w2, v2_z));
            // depth = _mm_div_ps(_mm_set1_ps(1.0f), depth);

            mask = _mm_and_si128(_mm_set1_epi32(1), mask);

            // Where are has alread been computed as (1/area)
            // texture from image
            //  u = (tex_w - 1) * (w0 * tex1[0] + w1 * tex2[0] + w2 * tex3[0]) * z
            //  v = (tex_h - 1) * (w0 * tex1[1] + w1 * tex2[1] + w2 * tex3[1]) * z
            const __m128 weighted_textures = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(_mm_mul_ps(w0, oneOverTriArea), texture1),
                    _mm_mul_ps(_mm_mul_ps(w1, oneOverTriArea), texture2)),
                _mm_mul_ps(_mm_mul_ps(w2, oneOverTriArea), texture3));

            __m128 tex_uv = _mm_set_ps(0.0f, 0.0f, (float)(tex_h - 1), (float)(tex_w - 1));
            tex_uv = _mm_mul_ps(tex_uv, weighted_textures);
            tex_uv = _mm_mul_ps(tex_uv, depth);

            float tex_coordinates[4];
            _mm_store_ps(tex_coordinates, tex_uv);

            unsigned char *texRGB = render->tex_data + ((int)tex_coordinates[0] + tex_w * (int)tex_coordinates[1]) * bpp;

            const SDL_Colour draw_colour_values = {texRGB[0], texRGB[1], texRGB[2], texRGB[3]};

            if (mask.m128i_i32[3])
                Draw_Pixel(render->fmt, render->pixels, x + 0, y, &draw_colour_values);

            if (mask.m128i_i32[2])
                Draw_Pixel(render->fmt, render->pixels, x + 1, y, &draw_colour_values);

            if (mask.m128i_i32[1])
                Draw_Pixel(render->fmt, render->pixels, x + 2, y, &draw_colour_values);

            if (mask.m128i_i32[0])
                Draw_Pixel(render->fmt, render->pixels, x + 3, y, &draw_colour_values);
        }

        // One row step
        w0_row = _mm_add_ps(w0_row, B12);
        w1_row = _mm_add_ps(w1_row, B20);
        w2_row = _mm_add_ps(w2_row, B01);
    }
}