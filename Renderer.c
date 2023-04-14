#include "Renderer.h"

Renderer SDL_Startup(const char *title, unsigned int width, unsigned int height)
{
    Renderer rend = {0};

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

    rend.running = true;
    rend.HEIGHT  = height;
    rend.WIDTH   = width;
    return rend;
}

void SDL_CleanUp(Renderer *renderer)
{
    if (renderer->window)
    {
        SDL_DestroyWindow(renderer->window);
        renderer->window = NULL;
    }

    SDL_Quit();
}

static inline void Draw_Pixel_RGBA(const Rendering_data *ren, int x, int y, uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
{
    const int index = (int)y * ren->screen_width + (int)x;

    ren->pixels[index] = (Uint32)((alpha << 24) + (red << 16) + (green << 8) + (blue << 0));
}

static void Draw_Pixel_SDL_Colour(const Rendering_data *ren, int x, int y, const SDL_Colour *col)
{
    const uint8_t red = col->r;
    const uint8_t gre = col->g;
    const uint8_t blu = col->b;
    const uint8_t alp = col->a;

    Draw_Pixel_RGBA(ren, x, y, red, gre, blu, alp);
}

// THE EXTREMELY FAST LINE ALGORITHM Variation E (Addition Fixed Point PreCalc)
// http://www.edepot.com/algorithm.html
static void Draw_Line(const Rendering_data *ren, int x, int y, int x2, int y2, const SDL_Colour *col)
{
    bool yLonger  = false;
    int  shortLen = y2 - y;
    int  longLen  = x2 - x;
    if (abs(shortLen) > abs(longLen))
    {
        int swap = shortLen;
        shortLen = longLen;
        longLen  = swap;
        yLonger  = true;
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
                Draw_Pixel_SDL_Colour(ren, j >> 16, y, col);
                j += decInc;
            }
            return;
        }
        longLen += y;
        for (int j = 0x8000 + (x << 16); y >= longLen; --y)
        {
            Draw_Pixel_SDL_Colour(ren, j >> 16, y, col);

            j -= decInc;
        }
        return;
    }

    if (longLen > 0)
    {
        longLen += x;
        for (int j = 0x8000 + (y << 16); x <= longLen; ++x)
        {
            Draw_Pixel_SDL_Colour(ren, x, j >> 16, col);

            j += decInc;
        }
        return;
    }
    longLen += x;
    for (int j = 0x8000 + (y << 16); x >= longLen; --x)
    {
        Draw_Pixel_SDL_Colour(ren, x, j >> 16, col);
        j -= decInc;
    }
}

void Draw_Triangle_Outline(const Rendering_data *ren, const __m128 *verticies, const SDL_Colour *col)
{
    float vert1[4];
    _mm_store_ps(vert1, verticies[0]);
    float vert2[4];
    _mm_store_ps(vert2, verticies[1]);
    float vert3[4];
    _mm_store_ps(vert3, verticies[2]);

    Draw_Line(ren, (int)vert1[0], (int)vert1[1], (int)vert2[0], (int)vert2[1], col);
    Draw_Line(ren, (int)vert2[0], (int)vert2[1], (int)vert3[0], (int)vert3[1], col);
    Draw_Line(ren, (int)vert3[0], (int)vert3[1], (int)vert1[0], (int)vert1[1], col);
}

void Draw_Depth_Buffer(const Rendering_data *render_data)
{
    const __m128 max_depth = _mm_set1_ps(render_data->max_depth_value);
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
            depthvalues        = _mm_div_ps(depthvalues, max_depth);
            // depthvalues = _mm_min_ps(_mm_set1_ps(1.0f), depthvalues);
            depthvalues = _mm_mul_ps(depthvalues, value_255);

            float shading[4];
            _mm_store_ps(shading, depthvalues);

            Draw_Pixel_RGBA(render_data, x + 0, y + 0, (uint8_t)shading[3], (uint8_t)shading[3], (uint8_t)shading[3], 255);
            Draw_Pixel_RGBA(render_data, x + 1, y + 0, (uint8_t)shading[2], (uint8_t)shading[2], (uint8_t)shading[2], 255);
            Draw_Pixel_RGBA(render_data, x + 0, y + 1, (uint8_t)shading[1], (uint8_t)shading[1], (uint8_t)shading[1], 255);
            Draw_Pixel_RGBA(render_data, x + 1, y + 1, (uint8_t)shading[0], (uint8_t)shading[0], (uint8_t)shading[0], 255);
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
    const __m128i vec1 = _mm_cvtps_epi32(v1);
    const __m128i vec2 = _mm_cvtps_epi32(v2);
    const __m128i vec3 = _mm_cvtps_epi32(v3);

    __m128i min_values = _mm_and_si128(_mm_max_epi32(_mm_min_epi32(_mm_min_epi32(vec1, vec2), vec3), _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFE));
    __m128i max_values = _mm_min_epi32(_mm_add_epi32(_mm_max_epi32(_mm_max_epi32(vec1, vec2), vec3), _mm_set1_epi32(1)), _mm_set_epi32(0, 0, screen_height - 1, screen_width - 1));

    // Returns {maxX, minX, maxY, minY}
    return _mm_unpacklo_epi32(max_values, min_values);
}

// void Textured_Shading(const Rendering_data *render, const __m128 *screen_space, const __m128 *world_space,
//                       const __m128 *w_values, const __m128 *normal_values, const __m128 texture_u, const __m128 texture_v,
//                       const __m128 surface_normal, const PointLight *light, const Mat4x4 TBN)
//{
//     const __m128 v0 = screen_space[2];
//     const __m128 v1 = screen_space[1];
//     const __m128 v2 = screen_space[0];

//    __m128 world_v0 = world_space[2];
//    __m128 world_v1 = world_space[1];
//    __m128 world_v2 = world_space[0];

//    __m128 w0 = w_values[2];
//    __m128 w1 = w_values[1];
//    __m128 w2 = w_values[0];

//    const __m128 normal1 = normal_values[2];
//    const __m128 normal2 = normal_values[1];
//    const __m128 normal3 = normal_values[0];

//    __m128 colour1 = _mm_setzero_ps();
//    __m128 colour2 = _mm_setzero_ps();
//    __m128 colour3 = _mm_setzero_ps();
//    if (render->shading == GOURAND)
//    {
//        colour1 = Calculate_Light(light->position, _mm_setzero_ps(), world_space[2], normal1, 0.3f, 0.5, 0.8f, 64, render->shading);
//        colour2 = Calculate_Light(light->position, _mm_setzero_ps(), world_space[1], normal2, 0.3f, 0.5, 0.8f, 64, render->shading);
//        colour3 = Calculate_Light(light->position, _mm_setzero_ps(), world_space[0], normal3, 0.3f, 0.5, 0.8f, 64, render->shading);
//    }

//    // used when checking if w0,w1,w2 is greater than 0;
//    const __m128i fxptZero = _mm_setzero_si128();

//    /* get the bounding box of the triangle */
//    union AABB_u aabb;
//    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, render->screen_width, render->screen_height));

//    // X and Y value setup
//    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
//    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
//    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

//    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
//    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
//    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

//    // Edge Setup
//    const __m128i A0 = _mm_sub_epi32(v1_y, v2_y); // A01 = [1].Y - [2].Y
//    const __m128i A1 = _mm_sub_epi32(v2_y, v0_y); // A12 = [2].Y - [0].Y
//    const __m128i A2 = _mm_sub_epi32(v0_y, v1_y); // A20 = [0].Y - [1].Y

//    const __m128i B0 = _mm_sub_epi32(v2_x, v1_x);
//    const __m128i B1 = _mm_sub_epi32(v0_x, v2_x);
//    const __m128i B2 = _mm_sub_epi32(v1_x, v0_x);

//    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v2_y), _mm_mullo_epi32(v2_x, v1_y));
//    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v0_y), _mm_mullo_epi32(v0_x, v2_y));
//    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v1_y), _mm_mullo_epi32(v1_x, v0_y));

//    __m128i triArea = _mm_mullo_epi32(B2, A1);
//    triArea         = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

//    // Skip triangle if area is zero
//    if (triArea.m128i_i32[0] <= 0)
//    {
//        return;
//    }

//    // const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
//    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

//    const __m128i aa0Inc = _mm_slli_epi32(A0, 1);
//    const __m128i aa1Inc = _mm_slli_epi32(A1, 1);
//    const __m128i aa2Inc = _mm_slli_epi32(A2, 1);

//    const __m128i bb0Inc = _mm_slli_epi32(B0, 1);
//    const __m128i bb1Inc = _mm_slli_epi32(B1, 1);
//    const __m128i bb2Inc = _mm_slli_epi32(B2, 1);

//    const __m128i colOffset = _mm_set_epi32(0, 1, 0, 1);
//    const __m128i rowOffset = _mm_set_epi32(0, 0, 1, 1);

//    const __m128i col    = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
//    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
//    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
//    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

//    __m128i row    = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
//    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
//    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
//    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

//    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
//    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
//    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

//    // Cast depth buffer to float
//    float *pDepthBuffer = render->z_buffer_array;
//    int    rowIdx       = (aabb.minY * render->screen_width + 2 * aabb.minX);

//    __m128 light_position, camera_position;
//    if (render->shading == NORMAL_MAPPING)
//    {
//        light_position  = Matrix_Multiply_Vector_SIMD(TBN.elements, light->position);  // Tangent light position
//        camera_position = Matrix_Multiply_Vector_SIMD(TBN.elements, _mm_setzero_ps()); // Tangent camera position

//        world_v0 = Matrix_Multiply_Vector_SIMD(TBN.elements, world_v0);
//        world_v1 = Matrix_Multiply_Vector_SIMD(TBN.elements, world_v1);
//        world_v2 = Matrix_Multiply_Vector_SIMD(TBN.elements, world_v2);
//    }
//    else
//    {
//        light_position  = light->position;
//        camera_position = _mm_setzero_ps();
//    }

//    const float ambient_strength = 0.8f;

//    // Rasterize
//    for (int y = aabb.minY; y < aabb.maxY; y += 2,
//             rowIdx += 2 * render->screen_width,
//             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
//             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
//             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
//    {
//        // Barycentric coordinates at start of row
//        int     index = rowIdx;
//        __m128i alpha = sum0Row;
//        __m128i beta  = sum1Row;
//        __m128i gama  = sum2Row;

//        for (int x = aabb.minX; x < aabb.maxX; x += 2,
//                 index += 4,
//                 alpha = _mm_add_epi32(alpha, aa0Inc),
//                 beta  = _mm_add_epi32(beta, aa1Inc),
//                 gama  = _mm_add_epi32(gama, aa2Inc))
//        {
//            // Test Pixel inside triangle
//            // __m128i mask = w0 | w1 | w2;
//            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
//            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, beta), gama));

//            // Early out if all of this quad's pixels are outside the triangle.
//            if (_mm_test_all_zeros(mask, mask))
//                continue;

//            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
//            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
//            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);

//            // Compute barycentric-interpolated depth
//            __m128 depth = _mm_mul_ps(w0_area, w0);
//            depth        = _mm_add_ps(depth, _mm_mul_ps(w1_area, w1));
//            depth        = _mm_add_ps(depth, _mm_mul_ps(w2_area, w2));
//            depth        = _mm_rcp_ps(depth);

//            //// DEPTH BUFFER
//            const __m128 previousDepthValue           = _mm_load_ps(&pDepthBuffer[index]);
//            const __m128 are_new_depths_less_than_old = _mm_cmplt_ps(depth, previousDepthValue);
//            const __m128 which_depths_should_be_drawn = _mm_and_ps(are_new_depths_less_than_old, _mm_cvtepi32_ps(mask));
//            const __m128 updated_depth_values         = _mm_blendv_ps(previousDepthValue, depth, which_depths_should_be_drawn);
//            _mm_store_ps(&pDepthBuffer[index], updated_depth_values);

//            const __m128i finalMask = _mm_cvtps_epi32(which_depths_should_be_drawn);

//            // Precalulate uv constants
//            const __m128 depth_w = _mm_mul_ps(depth, _mm_set1_ps((float)render->tex_w - 1));
//            const __m128 depth_h = _mm_mul_ps(depth, _mm_set1_ps((float)render->tex_h - 1));

//            if (finalMask.m128i_i32[3])
//            {
//                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[3]);
//                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[3]);
//                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[3]);

//                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[3], w1_area.m128_f32[3], w0_area.m128_f32[3]);

//                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(3, 3, 3, 3)));
//                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(3, 3, 3, 3)));

//                const int res_u = (int)hsum_ps_sse3(u);
//                const int res_v = (int)hsum_ps_sse3(v);

//                __m128 frag_colour;
//                if (render->shading == TEXTURED)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == TEXTURED_PHONG)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == NORMAL_MAPPING)
//                {
//                    const __m128 frag_position = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, world_v0),
//                            _mm_mul_ps(weight2, world_v1)),
//                        _mm_mul_ps(weight3, world_v2));

//                    const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;
//                    __m128               frag_normal    = _mm_set_ps(0.0f, normal_texture[2] / 256.0f, normal_texture[1] / 256.0f, normal_texture[0] / 256.0f);
//                    frag_normal                         = _mm_sub_ps(_mm_mul_ps(frag_normal, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
//                    frag_normal                         = Normalize_m128(frag_normal);

//                    frag_colour = Calculate_Light(light->position, _mm_setzero_ps(), frag_position, frag_normal, 0.8f, 1.0f, 0.2f, 64, render->shading);

//                    const unsigned char *pixelOffset    = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
//                    const __m128         texture_colour = _mm_set_ps(1.0f, pixelOffset[2] / 256.0f, pixelOffset[1] / 256.0f, pixelOffset[0] / 256.0f);

//                    frag_colour = _mm_add_ps(texture_colour, frag_colour);
//                }

//                const uint8_t red = (uint8_t)(frag_colour.m128_f32[0]);
//                const uint8_t gre = (uint8_t)(frag_colour.m128_f32[1]);
//                const uint8_t blu = (uint8_t)(frag_colour.m128_f32[2]);
//                const uint8_t alp = (uint8_t)(255);

//                Draw_Pixel_RGBA(render, x + 0, y + 0, red, gre, blu, alp);
//            }

//            if (finalMask.m128i_i32[2])
//            {
//                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[2]);
//                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[2]);
//                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[2]);

//                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[2], w1_area.m128_f32[2], w0_area.m128_f32[2]);

//                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(2, 2, 2, 2)));
//                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(2, 2, 2, 2)));

//                const int res_u = (int)hsum_ps_sse3(u);
//                const int res_v = (int)hsum_ps_sse3(v);

//                __m128 frag_colour;
//                if (render->shading == TEXTURED)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == TEXTURED_PHONG)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == NORMAL_MAPPING)
//                {
//                    const __m128 frag_position = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, world_v0),
//                            _mm_mul_ps(weight2, world_v1)),
//                        _mm_mul_ps(weight3, world_v2));

//                    const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;
//                    __m128               frag_normal    = _mm_set_ps(0.0f, normal_texture[2] / 256.0f, normal_texture[1] / 256.0f, normal_texture[0] / 256.0f);
//                    frag_normal                         = _mm_sub_ps(_mm_mul_ps(frag_normal, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
//                    frag_normal                         = Normalize_m128(frag_normal);

//                    frag_colour = Calculate_Light(light->position, _mm_setzero_ps(), frag_position, frag_normal, 0.8f, 1.0f, 0.2f, 64, render->shading);

//                    const unsigned char *pixelOffset    = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
//                    const __m128         texture_colour = _mm_set_ps(1.0f, pixelOffset[2] / 256.0f, pixelOffset[1] / 256.0f, pixelOffset[0] / 256.0f);

//                    frag_colour = _mm_add_ps(texture_colour, frag_colour);
//                }

//                const uint8_t red = (uint8_t)(frag_colour.m128_f32[0]);
//                const uint8_t gre = (uint8_t)(frag_colour.m128_f32[1]);
//                const uint8_t blu = (uint8_t)(frag_colour.m128_f32[2]);
//                const uint8_t alp = (uint8_t)(255);

//                Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);
//            }

//            if (finalMask.m128i_i32[1])
//            {
//                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[1]);
//                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[1]);
//                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[1]);

//                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[1], w1_area.m128_f32[1], w0_area.m128_f32[1]);

//                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(1, 1, 1, 1)));
//                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(1, 1, 1, 1)));

//                const int res_u = (int)hsum_ps_sse3(u);
//                const int res_v = (int)hsum_ps_sse3(v);

//                __m128 frag_colour;
//                if (render->shading == TEXTURED)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == TEXTURED_PHONG)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == NORMAL_MAPPING)
//                {
//                    const __m128 frag_position = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, world_v0),
//                            _mm_mul_ps(weight2, world_v1)),
//                        _mm_mul_ps(weight3, world_v2));

//                    const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;
//                    __m128               frag_normal    = _mm_set_ps(0.0f, normal_texture[2] / 256.0f, normal_texture[1] / 256.0f, normal_texture[0] / 256.0f);
//                    frag_normal                         = _mm_sub_ps(_mm_mul_ps(frag_normal, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
//                    frag_normal                         = Normalize_m128(frag_normal);

//                    frag_colour = Calculate_Light(light->position, _mm_setzero_ps(), frag_position, frag_normal, 0.8f, 1.0f, 0.2f, 64, render->shading);

//                    const unsigned char *pixelOffset    = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
//                    const __m128         texture_colour = _mm_set_ps(1.0f, pixelOffset[2] / 256.0f, pixelOffset[1] / 256.0f, pixelOffset[0] / 256.0f);

//                    frag_colour = _mm_add_ps(texture_colour, frag_colour);
//                }

//                const uint8_t red = (uint8_t)(frag_colour.m128_f32[0]);
//                const uint8_t gre = (uint8_t)(frag_colour.m128_f32[1]);
//                const uint8_t blu = (uint8_t)(frag_colour.m128_f32[2]);
//                const uint8_t alp = (uint8_t)(255);

//                Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);
//            }

//            if (finalMask.m128i_i32[0])
//            {
//                const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[0]);
//                const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[0]);
//                const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[0]);

//                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[0], w1_area.m128_f32[0], w0_area.m128_f32[0]);

//                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_shuffle_ps(depth_w, depth_w, _MM_SHUFFLE(0, 0, 0, 0)));
//                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_shuffle_ps(depth_h, depth_h, _MM_SHUFFLE(0, 0, 0, 0)));

//                const int res_u = (int)hsum_ps_sse3(u);
//                const int res_v = (int)hsum_ps_sse3(v);

//                __m128 frag_colour;
//                if (render->shading == TEXTURED)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == TEXTURED_PHONG)
//                {
//                    frag_colour = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, colour1),
//                            _mm_mul_ps(weight2, colour2)),
//                        _mm_mul_ps(weight3, colour3));
//                }
//                else if (render->shading == NORMAL_MAPPING)
//                {
//                    const __m128 frag_position = _mm_add_ps(
//                        _mm_add_ps(
//                            _mm_mul_ps(weight1, world_v0),
//                            _mm_mul_ps(weight2, world_v1)),
//                        _mm_mul_ps(weight3, world_v2));

//                    const unsigned char *normal_texture = render->nrm_data + (res_v + (render->nrm_w * res_u)) * render->nrm_bpp;
//                    __m128               frag_normal    = _mm_set_ps(0.0f, normal_texture[2] / 256.0f, normal_texture[1] / 256.0f, normal_texture[0] / 256.0f);
//                    frag_normal                         = _mm_sub_ps(_mm_mul_ps(frag_normal, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f));
//                    frag_normal                         = Normalize_m128(frag_normal);

//                    frag_colour = Calculate_Light(light->position, _mm_setzero_ps(), frag_position, frag_normal, 0.8f, 1.0f, 0.2f, 64, render->shading);

//                    const unsigned char *pixelOffset    = render->tex_data + (res_v + (render->tex_w * res_u)) * render->tex_bpp;
//                    const __m128         texture_colour = _mm_set_ps(1.0f, pixelOffset[2] / 256.0f, pixelOffset[1] / 256.0f, pixelOffset[0] / 256.0f);

//                    frag_colour = _mm_add_ps(texture_colour, frag_colour);
//                }

//                const uint8_t red = (uint8_t)(frag_colour.m128_f32[0]);
//                const uint8_t gre = (uint8_t)(frag_colour.m128_f32[1]);
//                const uint8_t blu = (uint8_t)(frag_colour.m128_f32[2]);
//                const uint8_t alp = (uint8_t)(255);

//                Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
//            }
//        }
//    }
//}

// Gouraud shading, which interpolates colors across polygons
static inline __m128 _Gourand_Shading_Get_Colour(const __m128 weights[3], const __m128 colours[3])
{
    return _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(weights[0], colours[0]),
            _mm_mul_ps(weights[1], colours[1])),
        _mm_mul_ps(weights[2], colours[2]));
}

// TODO: Pack the relevant light data into some struct
// TODO: Set a global render mode (global_renderer_state)
// Phong shading, a normal vector is linearly interpolated across the surface of the polygon from the polygon's vertex normals
static inline __m128 _Phong_Shading_Get_Colour(const __m128 weights[3], const __m128 colours[3], const __m128 world_space_coords[3],
                                               const __m128 normals[3], const __m128 camera_postion, const Light *light, const Shading_Mode shading_mode)
{
    const __m128 frag_position = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(weights[0], world_space_coords[0]),
            _mm_mul_ps(weights[1], world_space_coords[1])),
        _mm_mul_ps(weights[2], world_space_coords[2]));

    // Calculate the normal vector of the surface at the point, interpolating the surface normals at the vertices of the triangle
    const __m128 frag_normal = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(weights[0], normals[0]),
            _mm_mul_ps(weights[1], normals[1])),
        _mm_mul_ps(weights[2], normals[2]));

    return Light_Calculate_Shading(shading_mode, frag_position, frag_normal, camera_postion, light);
}

void Flat_Shading(const Rendering_data *render, const __m128 *screen_space_verticies, const __m128 *world_space_verticies, const float *w_values, const __m128 *normal_values, const __m128 camera_positon, const Light *light)
{
    const Shading_Mode shading_mode = GOURAND;

    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    // Unpack Vertex data
    const __m128 v0 = screen_space_verticies[2];
    const __m128 v1 = screen_space_verticies[1];
    const __m128 v2 = screen_space_verticies[0];

    // Gourand Shading
    // TODO: Fix this shading paramater passing BLIN_PHONG
    //__m128 colour1, colour2, colour3;
    __m128 colours[3];
    if (shading_mode == GOURAND) // We interpolate the colours in the "Vertex Shader"
    {
        colours[0] = Light_Calculate_Shading(shading_mode, world_space_verticies[0], normal_values[0], camera_positon, light);
        colours[1] = Light_Calculate_Shading(shading_mode, world_space_verticies[1], normal_values[1], camera_positon, light);
        colours[2] = Light_Calculate_Shading(shading_mode, world_space_verticies[2], normal_values[2], camera_positon, light);
    }
    else if (shading_mode == FLAT)
    {
        colours[0] = _mm_set_ps(255.f, 000.0f, 000.0f, 128.0f);
    }

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
    triArea         = _mm_sub_epi32(triArea, _mm_mullo_epi32(B1, A2));

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

    const __m128i col    = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

    __m128i row    = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));
    __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
    __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
    __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    // Cast depth buffer to float
    float *pDepthBuffer = render->z_buffer_array;
    int    rowIdx       = (aabb.minY * render->screen_width + 2 * aabb.minX);

    const __m128 one_over_w1 = _mm_set1_ps(w_values[0]);
    const __m128 one_over_w2 = _mm_set1_ps(w_values[1]);
    const __m128 one_over_w3 = _mm_set1_ps(w_values[2]);

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += 2 * render->screen_width,
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Barycentric coordinates at start of row
        int     index = rowIdx;
        __m128i alpha = sum0Row;
        __m128i beta  = sum1Row;
        __m128i gama  = sum2Row;

        for (int x = aabb.minX; x < aabb.maxX; x += 2,
                 index += 4,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 beta  = _mm_add_epi32(beta, aa1Inc),
                 gama  = _mm_add_epi32(gama, aa2Inc))
        {
            // Test Pixel inside triangle
            // __m128i mask = w0 | w1 | w2;
            // we compare < 0.0f, so we get all the values 0.0f and above, -1 values are "true"
            const __m128i mask = _mm_cmplt_epi32(fxptZero, _mm_or_si128(_mm_or_si128(alpha, beta), gama));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(w0_area, one_over_w1);
            depth        = _mm_add_ps(depth, _mm_mul_ps(w1_area, one_over_w2));
            depth        = _mm_add_ps(depth, _mm_mul_ps(w2_area, one_over_w3));
            depth        = _mm_rcp_ps(depth);

            //// DEPTH BUFFER
            const __m128 previousDepthValue           = _mm_load_ps(&pDepthBuffer[index]);
            const __m128 are_new_depths_less_than_old = _mm_cmplt_ps(depth, previousDepthValue);
            const __m128 which_depths_should_be_drawn = _mm_and_ps(are_new_depths_less_than_old, _mm_cvtepi32_ps(mask));
            const __m128 updated_depth_values         = _mm_blendv_ps(previousDepthValue, depth, which_depths_should_be_drawn);
            _mm_store_ps(&pDepthBuffer[index], updated_depth_values);

            const __m128i finalMask = _mm_cvtps_epi32(which_depths_should_be_drawn);

            // Loop over each pixel and draw
            for (int pixel_index = 0; pixel_index < 4; pixel_index++)
            {
                if (!finalMask.m128i_i32[pixel_index])
                    continue;

                __m128 frag_colour;
                if (shading_mode == FLAT)
                {
                    frag_colour = colours[0];
                }
                else
                {
                    // Interpolate shaded colour
                    const __m128 weights[3] = {
                        _mm_set1_ps(w0_area.m128_f32[pixel_index]),
                        _mm_set1_ps(w1_area.m128_f32[pixel_index]),
                        _mm_set1_ps(w2_area.m128_f32[pixel_index]),
                    };

                    // GOURAND Shading ------
                    if (shading_mode == GOURAND)
                    {
                        frag_colour = _Gourand_Shading_Get_Colour(weights, colours);
                    }
                    else if (shading_mode == PHONG)
                    {
                        frag_colour = _Phong_Shading_Get_Colour(weights, colours, world_space_verticies, normal_values, camera_positon, light, shading_mode);
                    }
                }

                const uint8_t red = (uint8_t)frag_colour.m128_f32[0];
                const uint8_t gre = (uint8_t)frag_colour.m128_f32[1];
                const uint8_t blu = (uint8_t)frag_colour.m128_f32[2];
                const uint8_t alp = (uint8_t)frag_colour.m128_f32[3];

                // Not sure if I like this
                if (pixel_index == 3) // index 3
                    Draw_Pixel_RGBA(render, x + 0, y + 0, red, gre, blu, alp);

                else if (pixel_index == 2) // index 2
                    Draw_Pixel_RGBA(render, x + 1, y + 0, red, gre, blu, alp);

                else if (pixel_index == 1) // index 1
                    Draw_Pixel_RGBA(render, x + 0, y + 1, red, gre, blu, alp);

                else if (pixel_index == 0) // index 0
                    Draw_Pixel_RGBA(render, x + 1, y + 1, red, gre, blu, alp);
            }
        }
    }
}