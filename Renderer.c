#include "Renderer.h"

Renderer global_renderer = {0};
AppState global_app      = {0}; // TODO: Rename this

void Reneder_Startup(const char *title, const int width, const int height)
{
    memset((void *)&global_renderer, 0, sizeof(Renderer));
    memset((void *)&global_app, 0, sizeof(AppState));

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        fprintf(stderr, "Could not SDL_Init(SDL_INIT_VIDEO): %s\n", SDL_GetError());
        exit(2);
    }

    global_renderer.window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_SHOWN); // show upon creation

    if (global_renderer.window == NULL)
    {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        exit(2);
    }

#ifdef GRAPHICS_USE_SDL_RENDERER
    SDL_Renderer *renderer   = SDL_CreateRenderer(global_renderer.window, -1, 0);
    global_renderer.renderer = renderer;

    SDL_Texture *texture    = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, width, height);
    global_renderer.texture = texture;

    global_renderer.screen_num_pixels = width * height;
#else
    SDL_Surface *window_surface = SDL_GetWindowSurface(global_renderer.window);
    global_renderer.surface     = window_surface;

    global_renderer.fmt = window_surface->format;

    printf("Window Surface\n\tPixel format : %s\n", SDL_GetPixelFormatName(global_renderer.surface->format->format));
    printf("\tBytesPP      : %d\n", global_renderer.fmt->BytesPerPixel);
    printf("\tBPP          : %d\n", global_renderer.fmt->BitsPerPixel);
    printf("\tPitch : %d\n", window_surface->pitch);

    // global_renderer.pixels            = (uint8_t *)window_surface->pixels;
    global_renderer.pixels            = (uint8_t *)window_surface->pixels;
    global_renderer.height            = window_surface->h;
    global_renderer.width             = window_surface->w;
    global_renderer.screen_num_pixels = window_surface->h * window_surface->w;
#endif

    // Allocate z buffer
    float *z_buff = (float *)_aligned_malloc(sizeof(float) * global_renderer.screen_num_pixels, 16);
    if (!z_buff)
    {
        fprintf(stderr, "Error with : (float *)_aligned_malloc(sizeof(float) * global_renderer.screen_num_pixels, 16);\n");
        exit(2);
    }
    global_renderer.z_buffer_array  = z_buff;
    global_renderer.max_depth_value = 50.0f;

    global_renderer.running = true;
}

void Renderer_Destroy(void)
{
    if (global_renderer.window)
    {
        SDL_DestroyWindow(global_renderer.window);
        global_renderer.window = NULL;
    }

#ifdef GRAPHICS_USE_SDL_RENDERER
    if (global_renderer.renderer)
    {
        SDL_DestroyRenderer(global_renderer.renderer);
        global_renderer.renderer = NULL;
    }
    if (global_renderer.texture)
    {
        SDL_DestroyTexture(global_renderer.texture);
        global_renderer.texture = NULL;
    }
#else
    if (global_renderer.surface)
    {
        SDL_FreeSurface(global_renderer.surface);
        global_renderer.surface = NULL;
        global_renderer.pixels  = NULL;
    }
    if (global_renderer.pixels)
    {
        free(global_renderer.pixels);
        global_renderer.pixels = NULL;
    }
#endif

    if (global_renderer.z_buffer_array)
    {
        _aligned_free(global_renderer.z_buffer_array);
        global_renderer.z_buffer_array = NULL;
    }

    SDL_Quit();

    fprintf(stderr, "Renderer has been destroyed\n");
}

static inline void Draw_Pixel_RGBA(const int x, const int y, const uint8_t red, const uint8_t green, const uint8_t blue, const uint8_t alpha)
{
    const int index = y * global_renderer.width + x;
    ASSERT(index <= global_renderer.screen_num_pixels);

    global_renderer.pixels[index * 4 + 0] = green;
    global_renderer.pixels[index * 4 + 1] = red;
    global_renderer.pixels[index * 4 + 2] = blue;
    global_renderer.pixels[index * 4 + 3] = 255;
}

static inline void Draw_Pixel_SDL_Colour(const int x, const int y, const SDL_Colour *col)
{
    Draw_Pixel_RGBA(x, y, col->r, col->g, col->b, col->a);
}

// THE EXTREMELY FAST LINE ALGORITHM Variation E (Addition Fixed Point PreCalc)
// http://www.edepot.com/algorithm.html
static void Draw_Line(int x, int y, int x2, int y2, const SDL_Colour *col)
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
                Draw_Pixel_SDL_Colour(j >> 16, y, col);
                j += decInc;
            }
            return;
        }
        longLen += y;
        for (int j = 0x8000 + (x << 16); y >= longLen; --y)
        {
            Draw_Pixel_SDL_Colour(j >> 16, y, col);

            j -= decInc;
        }
        return;
    }

    if (longLen > 0)
    {
        longLen += x;
        for (int j = 0x8000 + (y << 16); x <= longLen; ++x)
        {
            Draw_Pixel_SDL_Colour(x, j >> 16, col);

            j += decInc;
        }
        return;
    }
    longLen += x;
    for (int j = 0x8000 + (y << 16); x >= longLen; --x)
    {
        Draw_Pixel_SDL_Colour(x, j >> 16, col);
        j -= decInc;
    }
}

void Draw_Depth_Buffer(void)
{
    const __m128 max_depth = _mm_set1_ps(global_renderer.max_depth_value);
    const __m128 value_255 = _mm_set1_ps(255.0f);

    float *pDepthBuffer = global_renderer.z_buffer_array;
#if 1
    for (int y = 0; y < global_renderer.height; y++)
    {
        for (int x = 0; x < global_renderer.width; x += 4)
        {
            __m128 depthvalues = _mm_load_ps(&pDepthBuffer[y * global_renderer.width + x]);
            depthvalues        = _mm_mul_ps(_mm_add_ps(depthvalues, _mm_set1_ps(1.0f)), _mm_set1_ps(0.5f));
            depthvalues        = _mm_rcp_ps(depthvalues);
            depthvalues        = _mm_div_ps(depthvalues, max_depth);
            depthvalues        = _mm_mul_ps(depthvalues, value_255);

            float shading[4];
            _mm_store_ps(shading, depthvalues);

            Draw_Pixel_RGBA(x + 0, y, (uint8_t)shading[0], (uint8_t)shading[0], (uint8_t)shading[0], 255);
            Draw_Pixel_RGBA(x + 1, y, (uint8_t)shading[1], (uint8_t)shading[1], (uint8_t)shading[1], 255);
            Draw_Pixel_RGBA(x + 2, y, (uint8_t)shading[2], (uint8_t)shading[2], (uint8_t)shading[2], 255);
            Draw_Pixel_RGBA(x + 3, y, (uint8_t)shading[3], (uint8_t)shading[3], (uint8_t)shading[3], 255);
        }
    }
#else

    int rowIdx = 0;
    for (int y = 0; y < global_renderer.height; y += 2, rowIdx += 2 * global_renderer.width)
    {
        int index = rowIdx;
        for (int x = 0; x < global_renderer.width; x += 2, index += 4)
        {
            const __m128 depthVec = _mm_load_ps(&pDepthBuffer[index]);

            __m128 cmp = _mm_cmpeq_ps(depthVec, _mm_set1_ps(global_renderer.max_depth_value));

            if (_mm_testz_si128(_mm_cvtps_epi32(cmp), _mm_cvtps_epi32(cmp)) == 0)
                continue;

            // Normalize the depth values between 0 and 1
            __m128 normalizedDepthVec = _mm_div_ps(_mm_sub_ps(depthVec, minDepthVec), _mm_sub_ps(maxDepthVec, minDepthVec));

            // Map the normalized depth values to the range of 0 to 255
            __m128i colorValueVec = _mm_cvtps_epi32(_mm_mul_ps(normalizedDepthVec, _mm_set1_ps(255.0f)));

            // Convert the packed integer values to 8-bit unsigned integers
            __m128i colorValueU8Vec = _mm_packus_epi32(colorValueVec, colorValueVec);
            colorValueU8Vec         = _mm_packus_epi16(colorValueU8Vec, colorValueU8Vec);

            // Store the color values into the RGBA color buffer
            _mm_storeu_si128((__m128i *)&RenderState.colour_buffer[i], colorValueU8Vec);
        }
    }
#endif
}

/**
 * Calculates the color value of a pixel using Gourand shading.
 * The color value is determined by blending the colors of the surrounding vertices
 * based on their weights, which are calculated using barycentric coordinates.
 *
 * @param weights: the barycentric coordinates of the current fragment with respect to the triangle vertices
 * @param colours: colour at each vertex
 *
 * @return: the final colour of the fragment using the Gourand shading model
 */
static inline __m128 _Gourand_Shading_Get_Colour(const __m128 weights[3], const __m128 colours[3])
{
    return _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(weights[0], colours[0]),
            _mm_mul_ps(weights[1], colours[1])),
        _mm_mul_ps(weights[2], colours[2]));
}

/**
 * Calculate the final colour of the current fragment using Phong shading model. Phong shading, a normal vector is linearly
 * interpolated across the surface of the polygon from the polygon's vertex normals
 *
 * @param weights: the barycentric coordinates of the current fragment with respect to the triangle vertices
 * @param world_space_coords: the world space coordinates of the triangle vertices
 * @param normals: the surface normals at the triangle vertices
 * @param light: a pointer to the light object used in the shading calculation
 *
 * @return: the final colour of the fragment using the Phong shading model
 */
static inline __m128 _Phong_Shading_Get_Colour(const __m128 weights[3], const mvec4 world_space_coords[3], const mvec4 normals[3], const Light *light)
{
    const __m128 frag_position = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(weights[0], world_space_coords[0].m),
            _mm_mul_ps(weights[1], world_space_coords[1].m)),
        _mm_mul_ps(weights[2], world_space_coords[2].m));

    // Calculate the normal vector of the surface at the point, interpolating the surface normals at the vertices of the triangle
    const __m128 frag_normal = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(weights[0], normals[0].m),
            _mm_mul_ps(weights[1], normals[1].m)),
        _mm_mul_ps(weights[2], normals[2].m));

    // We should combine the lighting colour value and the interpolated vertex colours here...

    const mvec4 shading_colour = Light_Calculate_Shading((mvec4){.m = frag_position}, (mvec4){.m = frag_normal}, global_app.camera_position, light->position, light);

    return shading_colour.m;
}

/**
 * Get the normal vector from a normal map texture
 *
 * @param res_v: texture u coordinate in the texture space
 * @param res_u: texture v coordinate in the texture space
 *
 * @return: __m128 containing the colour as a vector in the range -1.0f to 1.0f
 */
static inline __m128 _Get_Normal_From_Normal_Map(const int res_v, const int res_u)
{
    const Texture_t normal_map = global_app.nrm;

    // Calculate the texture index in the normal map
    const unsigned int index = (res_u * normal_map.w + res_v) * normal_map.bpp;

    const unsigned char *normal_colour_data = normal_map.data + index;

    // Convert the RGB 0-255 values, to coordinates
    const float r = ((normal_colour_data[0] / 255.0f) * 2.0f) - 1.0f; // x
    const float g = ((normal_colour_data[1] / 255.0f) * 2.0f) - 1.0f; // y
    const float b = ((normal_colour_data[2] / 255.0f) * 2.0f) - 1.0f; // z

    mvec4 frag_normal = mate_vec4(r, g, b, 0.0f);
    frag_normal       = mate_norm(frag_normal);

    return frag_normal.m;
}

/**
 * Get the texture colour
 *
 * @param res_v: texture u coordinate in the texture space
 * @param res_u: texture v coordinate in the texture space
 *
 * @return: __m128 containing the colour in the range of 0.0f to 1.0f
 */
static inline __m128 _Get_Colour_From_Diffuse_Texture(const int res_v, const int res_u)
{
    const Texture_t texture = global_app.tex; // TODO : Remove this?

    ASSERT(res_u <= texture.w);
    ASSERT(res_v <= texture.h);
    ASSERT(texture.data);

    // Calculate the texture index in the texture image
    const unsigned int index = (res_u * texture.w + res_v) * texture.bpp;

    const unsigned char *pixel_data = texture.data + index;

    return _mm_div_ps(_mm_setr_ps(pixel_data[0], pixel_data[1], pixel_data[2], 255.0f),
                      _mm_set1_ps(255.0f));
}

#if 0 // Not feeling ready for this yet lol
void Textured_Shading(const __m128 *screen_space_verticies, const __m128 *world_space_verticies, const float *w_values, const __m128 *normal_values,
                      const __m128 texture_u, const __m128 texture_v, const mmat3 TBN, Light *light)
{
    // Unpack Vertex data
    const __m128 v0 = screen_space_verticies[2];
    const __m128 v1 = screen_space_verticies[1];
    const __m128 v2 = screen_space_verticies[0];

    __m128 world_v0 = world_space_verticies[2];
    __m128 world_v1 = world_space_verticies[1];
    __m128 world_v2 = world_space_verticies[0];

    __m128 light_position  = light->position;
    __m128 camera_position = global_app.camera_position;

    if (global_app.shading_mode == NORMAL_MAPPING)
    {
        /*
        Instead of sending the inverse of the TBN matrix to the "fragment shader",
        we send a tangent-space light position, camera position, and vertex position to the fragment shader.
        This saves us from having to do matrix multiplications in the fragment shader.

       TangentLightPos = TBN * light_position;
       TangentViewPos  = TBN * camera_position;
       TangentFragPos  = TBN * vec3(model * vec4(vert_pos, 1.0));

       vec3(model * vec4(vert_pos, 1.0)) < are the verts in world space
       */
        light_position  = Mat3x3_mul_m128(TBN, light->position);            // Tangent light position
        camera_position = Mat3x3_mul_m128(TBN, global_app.camera_position); // Tangent camera position

        world_v0 = Mat3x3_mul_m128(TBN, world_v0);
        world_v1 = Mat3x3_mul_m128(TBN, world_v1);
        world_v2 = Mat3x3_mul_m128(TBN, world_v2);
    }

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, global_renderer.width, global_renderer.height));

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
    float *pDepthBuffer = global_renderer.z_buffer_array;
    int    rowIdx       = (aabb.minY * global_renderer.width + 2 * aabb.minX);

    const __m128 one_over_w1 = _mm_set1_ps(w_values[2]);
    const __m128 one_over_w2 = _mm_set1_ps(w_values[1]);
    const __m128 one_over_w3 = _mm_set1_ps(w_values[0]);

    // Generate masks used for tie-breaking rules (not to double-shade along shared edges)
    // there is no _mm_cmpge_epi32, so use lt and swap operands
    // _mm_cmplt_epi32(bb0Inc, _mm_setzero_si128()) - becomes - _mm_cmplt_epi32(_mm_setzero_si128(), bb0Inc)
    const __m128i Edge0TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(aa0Inc, _mm_setzero_si128()),
                                               _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), bb0Inc), _mm_cmpeq_epi32(aa0Inc, _mm_setzero_si128())));

    const __m128i Edge1TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(aa1Inc, _mm_setzero_si128()),
                                               _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), bb1Inc), _mm_cmpeq_epi32(aa1Inc, _mm_setzero_si128())));

    const __m128i Edge2TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(aa2Inc, _mm_setzero_si128()),
                                               _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), bb2Inc), _mm_cmpeq_epi32(aa2Inc, _mm_setzero_si128())));

    // Rasterize
    const int row_index_step_amount = 2 * global_renderer.width;
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += row_index_step_amount,
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
            // "FRAGMENT SHADER"

            // Test Pixel inside triangle
            // const __m128i sseEdge0Positive = _mm_cmpgt_epi32(alpha, _mm_setzero_si128());
            // const __m128i sseEdge0Negative = _mm_cmplt_epi32(alpha, _mm_setzero_si128());
            // const __m128i sseEdge0FuncMask = _mm_or_epi32(sseEdge0Positive,
            //                                              _mm_andnot_epi32(sseEdge0Negative, Edge0TieBreak));

            //// Edge 1 test
            // const __m128i sseEdge1Positive = _mm_cmpgt_epi32(beta, _mm_setzero_si128());
            // const __m128i sseEdge1Negative = _mm_cmplt_epi32(beta, _mm_setzero_si128());
            // const __m128i sseEdge1FuncMask = _mm_or_epi32(sseEdge1Positive,
            //                                               _mm_andnot_epi32(sseEdge1Negative, Edge1TieBreak));

            //// Edge 2 test
            // const __m128i sseEdge2Positive = _mm_cmpgt_epi32(gama, _mm_setzero_si128());
            // const __m128i sseEdge2Negative = _mm_cmplt_epi32(gama, _mm_setzero_si128());
            // const __m128i sseEdge2FuncMask = _mm_or_epi32(sseEdge2Positive,
            //                                               _mm_andnot_epi32(sseEdge2Negative, Edge2TieBreak));

            // Combine resulting masks of all three edges
            // const __m128i mask = _mm_and_epi32(sseEdge0FuncMask,
            //                                   _mm_and_epi32(sseEdge1FuncMask, sseEdge2FuncMask));

            const __m128i mask = _mm_cmplt_epi32(_mm_setzero_si128(), _mm_or_si128(_mm_or_si128(alpha, beta), gama));

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0_area = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1_area = _mm_mul_ps(_mm_cvtepi32_ps(beta), oneOverTriArea);
            const __m128 w2_area = _mm_mul_ps(_mm_cvtepi32_ps(gama), oneOverTriArea);

            // Compute barycentric-interpolated depth
            __m128 depth = _mm_mul_ps(w0_area, one_over_w1);
            depth        = _mm_add_ps(depth, _mm_mul_ps(w1_area, one_over_w2));
            depth        = _mm_add_ps(depth, _mm_mul_ps(w2_area, one_over_w3));

            __m128 inv_depth = _mm_rcp_ps(depth); // 1.0 / depth

            //// DEPTH BUFFER
            const __m128 previousDepthValue           = _mm_load_ps(&pDepthBuffer[index]);
            const __m128 are_new_depths_less_than_old = _mm_cmplt_ps(inv_depth, previousDepthValue);
            const __m128 which_depths_should_be_drawn = _mm_and_ps(are_new_depths_less_than_old, _mm_cvtepi32_ps(mask));
            const __m128 updated_depth_values         = _mm_blendv_ps(previousDepthValue, inv_depth, which_depths_should_be_drawn);
            _mm_store_ps(&pDepthBuffer[index], updated_depth_values);

            const __m128i finalMask = _mm_cvtps_epi32(which_depths_should_be_drawn);


            // Precalulate uv constants
            const __m128 depth_w = _mm_mul_ps(inv_depth, _mm_set1_ps((float)global_app.tex.w - 1.0f));
            const __m128 depth_h = _mm_mul_ps(inv_depth, _mm_set1_ps((float)global_app.tex.h - 1.0f));
            for (int pixel_index = 0; pixel_index < 4; pixel_index++)
            {
                if (!finalMask.m128i_i32[pixel_index])
                    continue;

                const __m128 weights = _mm_set_ps(0.0f, w2_area.m128_f32[pixel_index], w1_area.m128_f32[pixel_index], w0_area.m128_f32[pixel_index]);

                const __m128 u = _mm_mul_ps(_mm_mul_ps(texture_u, weights), _mm_set1_ps(depth_w.m128_f32[pixel_index]));
                const __m128 v = _mm_mul_ps(_mm_mul_ps(texture_v, weights), _mm_set1_ps(depth_h.m128_f32[pixel_index]));

                const int res_u = (int)hsum_ps_sse3(u);
                const int res_v = (int)hsum_ps_sse3(v);

                __m128 frag_colour = {0};
                if (global_app.shading_mode == TEXTURED)
                {
                    frag_colour = _Get_Colour_From_Diffuse_Texture(res_v, res_u);
                }
                else if (global_app.shading_mode == TEXTURED_PHONG)
                {
                    const __m128 all_weights[3] = {
                        _mm_set1_ps(w2_area.m128_f32[pixel_index]),
                        _mm_set1_ps(w1_area.m128_f32[pixel_index]),
                        _mm_set1_ps(w0_area.m128_f32[pixel_index]),
                    };
                    // Calculate lighting
                    const __m128 lighting_contribution = _Phong_Shading_Get_Colour(all_weights, world_space_verticies, normal_values, light);

                    // Combine the lighting and texture colour together
                    const __m128 diff_colour = _Get_Colour_From_Diffuse_Texture(res_v, res_u);

                    frag_colour = _mm_mul_ps(diff_colour, lighting_contribution);
                }
                else if (global_app.shading_mode == NORMAL_MAPPING)
                {
                    const __m128 weight1 = _mm_set1_ps(w0_area.m128_f32[pixel_index]);
                    const __m128 weight2 = _mm_set1_ps(w1_area.m128_f32[pixel_index]);
                    const __m128 weight3 = _mm_set1_ps(w2_area.m128_f32[pixel_index]);

                    const __m128 frag_position = _mm_add_ps(
                        _mm_add_ps(
                            _mm_mul_ps(weight1, world_v0),
                            _mm_mul_ps(weight2, world_v1)),
                        _mm_mul_ps(weight3, world_v2));

                    const __m128 frag_normal = _Get_Normal_From_Normal_Map(res_v, res_u);

                    // Get the diffuse colour from the diffuse texture
                    const __m128 diff_colour = _Get_Colour_From_Diffuse_Texture(res_v, res_u);
                    light->diffuse_colour    = diff_colour;

                    // Calculate lighting
                    const __m128 lighting_contribution = Light_Calculate_Shading(frag_position, frag_normal, camera_position, light_position, light);

                    // frag_colour = _mm_mul_ps(_mm_set1_ps(1.0f), lighting_contribution);
                    frag_colour = _mm_mul_ps(_mm_set1_ps(255.0f), lighting_contribution);
                }

                const uint8_t red = (uint8_t)(frag_colour.m128_f32[0] * 255.0f);
                const uint8_t gre = (uint8_t)(frag_colour.m128_f32[1] * 255.0f);
                const uint8_t blu = (uint8_t)(frag_colour.m128_f32[2] * 255.0f);
                const uint8_t alp = (uint8_t)255;

                // Not sure if I like this
                if (pixel_index == 3) // index 3
                    Draw_Pixel_RGBA(x + 0, y + 0, red, gre, blu, alp);

                else if (pixel_index == 2) // index 2
                    Draw_Pixel_RGBA(x + 1, y + 0, red, gre, blu, alp);

                else if (pixel_index == 1) // index 1
                    Draw_Pixel_RGBA(x + 0, y + 1, red, gre, blu, alp);

                else if (pixel_index == 0) // index 0
                    Draw_Pixel_RGBA(x + 1, y + 1, red, gre, blu, alp);
            }
        }
    }
}
#endif

// NOTE : Temporary
static inline float hsum_ps_sse3(const __m128 v)
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void Flat_Shading(const RasterData_t rd)
{
    const __m128i colOffset = _mm_setr_epi32(0, 1, 2, 3); // NOTE: The "r" here!! makes loading and storing colour easier
    const __m128i rowOffset = _mm_setr_epi32(0, 0, 0, 0);

    if (global_app.shading_mode == WIRE_FRAME)
    {
        const float *v0 = rd.screen_space_verticies[0].f;
        const float *v1 = rd.screen_space_verticies[1].f;
        const float *v2 = rd.screen_space_verticies[2].f;

        const SDL_Colour col = {255, 255, 255, 255};
        Draw_Line((int)v0[0], (int)v0[1], (int)v1[0], (int)v1[1], &col);
        Draw_Line((int)v1[0], (int)v1[1], (int)v2[0], (int)v2[1], &col);
        Draw_Line((int)v2[0], (int)v2[1], (int)v0[0], (int)v0[1], &col);
        return;
    }

    // Unpack Vertex data
    const __m128 v0 = rd.screen_space_verticies[0].m;
    const __m128 v1 = rd.screen_space_verticies[1].m;
    const __m128 v2 = rd.screen_space_verticies[2].m;

    Texture_t tex = global_app.tex;

    // Setup for textured shading
    __m128 U[3], V[3], Z[3];
    for (size_t i = 0; i < 3; i++)
    {
        U[i] = _mm_set1_ps(rd.tex_u.f[i]);
        V[i] = _mm_set1_ps(rd.tex_v.f[i]);
        Z[i] = _mm_set1_ps(rd.screen_space_verticies[i].f[2]);
    }

    // Gourand Shading
    __m128 colours[3];
    if (global_app.shading_mode == GOURAND) // We interpolate the colours in the "Vertex Shader"
    {
        mvec4 tmp_colours0 = Light_Calculate_Shading(rd.world_space_verticies[0], rd.normals[0], global_app.camera_position, rd.light->position, rd.light);
        mvec4 tmp_colours1 = Light_Calculate_Shading(rd.world_space_verticies[1], rd.normals[1], global_app.camera_position, rd.light->position, rd.light);
        mvec4 tmp_colours2 = Light_Calculate_Shading(rd.world_space_verticies[2], rd.normals[2], global_app.camera_position, rd.light->position, rd.light);

        colours[0] = tmp_colours0.m;
        colours[1] = tmp_colours1.m;
        colours[2] = tmp_colours2.m;

        // We should combine the lighting colour value and the interpolated vertex colours here...
    }
    else if (global_app.shading_mode == FLAT)
    {
        colours[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f);
    }

    /* get the bounding box of the triangle */
    union AABB_u aabb;
    _mm_storeu_si128((__m128i *)aabb.values, Get_AABB_SIMD(v0, v1, v2, global_renderer.width, global_renderer.height));

    // X and Y value setup
    const __m128i v0_x = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v1_x = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 0)));
    const __m128i v2_x = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)));

    const __m128i v0_y = _mm_cvtps_epi32(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v1_y = _mm_cvtps_epi32(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128i v2_y = _mm_cvtps_epi32(_mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1)));

    // Edge Setup
    const __m128i A0 = _mm_sub_epi32(v2_y, v1_y);
    const __m128i A1 = _mm_sub_epi32(v0_y, v2_y);
    const __m128i A2 = _mm_sub_epi32(v1_y, v0_y);

    const __m128i B0 = _mm_sub_epi32(v1_x, v2_x);
    const __m128i B1 = _mm_sub_epi32(v2_x, v0_x);
    const __m128i B2 = _mm_sub_epi32(v0_x, v1_x);

    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(v2_x, v1_y), _mm_mullo_epi32(v1_x, v2_y));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(v0_x, v2_y), _mm_mullo_epi32(v2_x, v0_y));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(v1_x, v0_y), _mm_mullo_epi32(v0_x, v1_y));

    // Pass in Area?
    const __m128i triArea = _mm_sub_epi32(_mm_mullo_epi32(B2, A1), _mm_mullo_epi32(B1, A2));
    // const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
    const __m128 oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));

    const __m128i aa0Inc = _mm_slli_epi32(A0, 2);
    const __m128i aa1Inc = _mm_slli_epi32(A1, 2);
    const __m128i aa2Inc = _mm_slli_epi32(A2, 2);

    const __m128i bb0Inc = B0;
    const __m128i bb1Inc = B1;
    const __m128i bb2Inc = B2;

    const __m128i col = _mm_add_epi32(colOffset, _mm_set1_epi32(aabb.minX));
    const __m128i row = _mm_add_epi32(rowOffset, _mm_set1_epi32(aabb.minY));

    const __m128i aa0Col = _mm_mullo_epi32(A0, col);
    const __m128i aa1Col = _mm_mullo_epi32(A1, col);
    const __m128i aa2Col = _mm_mullo_epi32(A2, col);

    const __m128i bb0Row = _mm_add_epi32(_mm_mullo_epi32(B0, row), C0);
    const __m128i bb1Row = _mm_add_epi32(_mm_mullo_epi32(B1, row), C1);
    const __m128i bb2Row = _mm_add_epi32(_mm_mullo_epi32(B2, row), C2);

    __m128i sum0Row = _mm_add_epi32(aa0Col, bb0Row);
    __m128i sum1Row = _mm_add_epi32(aa1Col, bb1Row);
    __m128i sum2Row = _mm_add_epi32(aa2Col, bb2Row);

    // Cast depth buffer to float
    float   *pDepthBuffer  = global_renderer.z_buffer_array;
    uint8_t *pColourBuffer = global_renderer.pixels;

    const __m128 one_over_w0 = _mm_set1_ps(rd.w_values[0]);
    const __m128 one_over_w1 = _mm_set1_ps(rd.w_values[1]);
    const __m128 one_over_w2 = _mm_set1_ps(rd.w_values[2]);

    // Generate masks used for tie-breaking rules (not to double-shade along shared edges)
    // there is no _mm_cmpge_epi32, so use lt and swap operands
    // _mm_cmplt_epi32(bb0Inc, _mm_setzero_si128()) - becomes - _mm_cmplt_epi32(_mm_setzero_si128(), bb0Inc)
    const __m128i Edge0TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(aa0Inc, _mm_setzero_si128()),
                                               _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), bb0Inc), _mm_cmpeq_epi32(aa0Inc, _mm_setzero_si128())));

    const __m128i Edge1TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(aa1Inc, _mm_setzero_si128()),
                                               _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), bb1Inc), _mm_cmpeq_epi32(aa1Inc, _mm_setzero_si128())));

    const __m128i Edge2TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(aa2Inc, _mm_setzero_si128()),
                                               _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), bb2Inc), _mm_cmpeq_epi32(aa2Inc, _mm_setzero_si128())));

    // Rasterize
    for (int y       = aabb.minY; y <= aabb.maxY; y += 1,
             sum0Row = _mm_add_epi32(sum0Row, bb0Inc),
             sum1Row = _mm_add_epi32(sum1Row, bb1Inc),
             sum2Row = _mm_add_epi32(sum2Row, bb2Inc))
    {
        // Barycentric coordinates at start of row
        __m128i alpha = sum0Row;
        __m128i betaa = sum1Row;
        __m128i gamma = sum2Row;

        for (int x     = aabb.minX; x <= aabb.maxX; x += 4,
                 alpha = _mm_add_epi32(alpha, aa0Inc),
                 betaa = _mm_add_epi32(betaa, aa1Inc),
                 gamma = _mm_add_epi32(gamma, aa2Inc))
        {
            // Test Pixel inside triangle
            const __m128i sseEdge0Positive = _mm_cmpgt_epi32(alpha, _mm_setzero_si128());
            const __m128i sseEdge0Negative = _mm_cmplt_epi32(alpha, _mm_setzero_si128());
            const __m128i sseEdge0FuncMask = _mm_or_epi32(sseEdge0Positive,
                                                          _mm_andnot_epi32(sseEdge0Negative, Edge0TieBreak));

            // Edge 1 test
            const __m128i sseEdge1Positive = _mm_cmpgt_epi32(betaa, _mm_setzero_si128());
            const __m128i sseEdge1Negative = _mm_cmplt_epi32(betaa, _mm_setzero_si128());
            const __m128i sseEdge1FuncMask = _mm_or_epi32(sseEdge1Positive,
                                                          _mm_andnot_epi32(sseEdge1Negative, Edge1TieBreak));

            // Edge 2 test
            const __m128i sseEdge2Positive = _mm_cmpgt_epi32(gamma, _mm_setzero_si128());
            const __m128i sseEdge2Negative = _mm_cmplt_epi32(gamma, _mm_setzero_si128());
            const __m128i sseEdge2FuncMask = _mm_or_epi32(sseEdge2Positive,
                                                          _mm_andnot_epi32(sseEdge2Negative, Edge2TieBreak));

            // Combine resulting masks of all three edges
            const __m128i mask = _mm_and_epi32(sseEdge0FuncMask, _mm_and_epi32(sseEdge1FuncMask, sseEdge2FuncMask));

            // const __m128i mask = _mm_cmpgt_epi32(_mm_or_si128(_mm_or_si128(alpha, betaa), gamma), _mm_setzero_si128());

            // Early out if all of this quad's pixels are outside the triangle.
            if (_mm_test_all_zeros(mask, mask))
                continue;

            const __m128 w0 = _mm_mul_ps(_mm_cvtepi32_ps(alpha), oneOverTriArea);
            const __m128 w1 = _mm_mul_ps(_mm_cvtepi32_ps(betaa), oneOverTriArea);
            const __m128 w2 = _mm_mul_ps(_mm_cvtepi32_ps(gamma), oneOverTriArea);

            const __m128 weights[3] = {
                _mm_mul_ps(one_over_w0, w0),
                _mm_mul_ps(one_over_w1, w1),
                _mm_mul_ps(one_over_w2, w2),
            };

            __m128 intrFactor = _mm_add_ps(_mm_add_ps(weights[0], weights[1]), weights[2]);
            intrFactor        = _mm_rcp_ps(intrFactor);

            // Compute barycentric-interpolated depth
            // https://stackoverflow.com/questions/74261146/why-depth-values-must-be-interpolated-directly-by-barycentric-coordinates-in-ope
            __m128 depth = _mm_add_ps(_mm_add_ps(_mm_mul_ps(Z[0], w0), _mm_mul_ps(Z[1], w1)), _mm_mul_ps(Z[2], w2));
            // depth        = _mm_mul_ps(intrFactor, depth);

            const size_t index  = y * global_renderer.width + x;
            float       *pDepth = &pDepthBuffer[index];

            //// DEPTH BUFFER
            const __m128 previousDepthValue           = _mm_load_ps(pDepth);
            const __m128 are_new_depths_less_than_old = _mm_cmplt_ps(depth, previousDepthValue);

            if ((uint16_t)_mm_movemask_ps(are_new_depths_less_than_old) == 0x0000)
                continue;

            const __m128 which_depths_should_be_drawn = _mm_and_ps(are_new_depths_less_than_old, _mm_castsi128_ps(mask));
            const __m128 updated_depth_values         = _mm_blendv_ps(previousDepthValue, depth, which_depths_should_be_drawn);
            _mm_store_ps(pDepth, updated_depth_values);

            const __m128i finalMask = _mm_castps_si128(which_depths_should_be_drawn);

            intrFactor = _mm_mul_ps(intrFactor, _mm_cvtepi32_ps(_mm_abs_epi32(finalMask)));

            // Loop over each pixel and draw
            __m128i combined_colours = {0};
            if (global_app.shading_mode == SHADING_DEPTH_BUFFER)
            {
                continue;
            }
            else if (global_app.shading_mode == FLAT)
            {
                combined_colours = _mm_set1_epi32(0x00FF00FF);
            }
            else if (global_app.shading_mode == TEXTURED)
            {
                __m128 U_w = _mm_add_ps(_mm_add_ps(_mm_mul_ps(U[0], w0), _mm_mul_ps(U[1], w1)), _mm_mul_ps(U[2], w2));
                __m128 V_w = _mm_add_ps(_mm_add_ps(_mm_mul_ps(V[0], w0), _mm_mul_ps(V[1], w1)), _mm_mul_ps(V[2], w2));

                // clamp the vector to the range [0.0f, 1.0f]
                // U_w = _mm_max_ps(_mm_min_ps(U_w, _mm_set1_ps(1.0f)), _mm_setzero_ps());
                // V_w = _mm_max_ps(_mm_min_ps(V_w, _mm_set1_ps(1.0f)), _mm_setzero_ps());

                U_w = _mm_mul_ps(intrFactor, _mm_mul_ps(U_w, _mm_set1_ps((float)tex.w - 1)));
                V_w = _mm_mul_ps(intrFactor, _mm_mul_ps(V_w, _mm_set1_ps((float)tex.h - 1)));

                // (U + texture.width * V) * texture.bpp
                const __m128i texture_offset = _mm_mullo_epi32(
                    _mm_set1_epi32(tex.bpp),
                    _mm_add_epi32(
                        _mm_cvtps_epi32(U_w),
                        _mm_mullo_epi32(
                            _mm_set1_epi32(tex.w),
                            _mm_cvtps_epi32(V_w))));

                __m128i pixel_colour[4] = {0};
                for (int i = 0; i < 4; ++i)
                    pixel_colour[i] = _mm_loadu_si128((__m128i *)(tex.data + texture_offset.m128i_u32[i]));

                const __m128i interleaved1 = _mm_unpacklo_epi32(pixel_colour[0], pixel_colour[1]);
                const __m128i interleaved2 = _mm_unpacklo_epi32(pixel_colour[2], pixel_colour[3]);

                // combined: [r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3, r4, g4, b4, a4]
                combined_colours = _mm_unpacklo_epi64(interleaved1, interleaved2);
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
                    if (global_app.shading_mode == GOURAND)
                    {
                        colour = _Gourand_Shading_Get_Colour(weights, colours);
                    }
                    else if (global_app.shading_mode == PHONG || global_app.shading_mode == BLIN_PHONG)
                    {
                        colour = _Phong_Shading_Get_Colour(weights, rd.world_space_verticies, rd.normals, rd.light);
                    }
                }

                //__m128i pixel_colour[4] = {0};
                // for (int pixel_index = 0; pixel_index < 4; pixel_index++)
                //{
                //    if (finalMask.m128i_i32[pixel_index] == 0)
                //        continue;

                //    __m128 colour = {0};

                //    else if (global_app.shading_mode == PHONG || global_app.shading_mode == BLIN_PHONG)
                //    {
                //        combined_colours = _mm_set1_epi32(0x00FF00FF);
                //        // colour = _Phong_Shading_Get_Colour(weights, rd.world_space_verticies, rd.normals, rd.light);
                //    }

                //    pixel_colour[pixel_index] = _mm_cvtps_epi32(_mm_mul_ps(colour, _mm_set1_ps(255.0f)));
                //}
            }

            // Shuffle RGBA to BRGA
            combined_colours = _mm_shuffle_epi8(combined_colours, _mm_setr_epi8(
                                                                      1, 0, 2, 3,    // pix 1
                                                                      5, 4, 6, 7,    // pix 2
                                                                      9, 8, 10, 11,  // pix 3
                                                                      13, 12, 14, 15 // pix 4
                                                                      ));

            uint8_t *pixel_location = &pColourBuffer[index * 4];

#if 1 /* Fabian method */
            // We need to combine original pixel colour otherwise we would overrwite it to black lel
            const __m128i original_pixel_data = _mm_loadu_epi8(pixel_location);

            const __m128i masked_output = _mm_or_si128(_mm_and_si128(finalMask, combined_colours),
                                                       _mm_andnot_si128(finalMask, original_pixel_data));

            _mm_storeu_si128((__m128i *)pixel_location, masked_output);
#else
            // Mask-store 4-sample fragment values
            _mm_maskstore_epi32(
                (int *)pixel_location,
                finalMask,
                combined_colours);
#endif
        }
    }
}