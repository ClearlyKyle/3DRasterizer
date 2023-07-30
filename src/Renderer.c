#include "Renderer.h"

Renderer global_renderer = {0};

void Reneder_Startup(const char *title, const int width, const int height)
{
    memset((void *)&global_renderer, 0, sizeof(Renderer));

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
    printf("\tPitch        : %d\n", window_surface->pitch);

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
    global_renderer.max_depth_value = 100.0f;

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
    ASSERT(index >= 0 && index <= global_renderer.screen_num_pixels);

    global_renderer.pixels[index * 4 + 0] = green;
    global_renderer.pixels[index * 4 + 1] = red;
    global_renderer.pixels[index * 4 + 2] = blue;
    global_renderer.pixels[index * 4 + 3] = alpha;
}

static inline void Draw_Pixel_SDL_Colour(const int x, const int y, const SDL_Colour *col)
{
    Draw_Pixel_RGBA(x, y, col->r, col->g, col->b, col->a);
}

// http://www.edepot.com/algorithm.html
static void _draw_line(const int x, const int y, const int x2, const int y2, const SDL_Colour *col)
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

    const int incrementVal = (longLen < 0) ? -1 : 1;

    double multDiff = (longLen == 0.0) ? (double)shortLen : (double)shortLen / (double)longLen;

    if (yLonger)
    {
        for (int i = 0; i != longLen; i += incrementVal)
        {
            Draw_Pixel_SDL_Colour(x + (int)((double)i * multDiff), y + i, col);
        }
    }
    else
    {
        for (int i = 0; i != longLen; i += incrementVal)
        {
            Draw_Pixel_SDL_Colour(x + i, y + (int)((double)i * multDiff), col);
        }
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
            depthvalues        = _mm_div_ps(depthvalues, max_depth);
            // depthvalues        = _mm_mul_ps(_mm_add_ps(depthvalues, _mm_set1_ps(1.0f)), _mm_set1_ps(0.5f));
            // depthvalues = _mm_rcp_ps(depthvalues);
            depthvalues = _mm_mul_ps(depthvalues, value_255);

            float shading[4];
            _mm_store_ps(shading, depthvalues);

            Draw_Pixel_RGBA(x + 0, y, (uint8_t)shading[0], (uint8_t)shading[0], (uint8_t)shading[0], 255);
            Draw_Pixel_RGBA(x + 1, y, (uint8_t)shading[1], (uint8_t)shading[1], (uint8_t)shading[1], 255);
            Draw_Pixel_RGBA(x + 2, y, (uint8_t)shading[2], (uint8_t)shading[2], (uint8_t)shading[2], 255);
            Draw_Pixel_RGBA(x + 3, y, (uint8_t)shading[3], (uint8_t)shading[3], (uint8_t)shading[3], 255);
        }
    }

    // float whats_min_depth = global_renderer.max_depth_value;
    // float whats_max_depth = 0.0f;
    // for (size_t i = 0; i < global_renderer.height * global_renderer.width; i++)
    //{
    //     const float val = pDepthBuffer[i];

    //    if (val == global_renderer.max_depth_value)
    //        continue;

    //    whats_max_depth = val > whats_max_depth ? val : whats_max_depth;
    //    whats_min_depth = val < whats_min_depth ? val : whats_min_depth;
    //}
    // printf("Depth - min : %f, max : %f\n", whats_min_depth, whats_max_depth);
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

/*
    dest[0] = weights[0] * attibute[0].X + weights[1] * attibute[1].X + weights[2] * attibute[2].X
    dest[1] = weights[0] * attibute[0].Y + weights[1] * attibute[1].Y + weights[2] * attibute[2].Y
    dest[2] = weights[0] * attibute[0].Z + weights[1] * attibute[1].Z + weights[2] * attibute[2].Z
    dest[3] = weights[0] * attibute[0].W + weights[1] * attibute[1].W + weights[2] * attibute[2].W
*/
static inline void _Interpolate_Something(const __m128 persp[3], const __m128 attribues[3], __m128 dest[4])
{
    const __m128 X0 = _mm_shuffle_ps(attribues[0], attribues[0], _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 X1 = _mm_shuffle_ps(attribues[1], attribues[1], _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 X2 = _mm_shuffle_ps(attribues[2], attribues[2], _MM_SHUFFLE(0, 0, 0, 0));

    __m128 preX[3];
    preX[0] = _mm_mul_ps(X0, persp[0]);
    preX[1] = _mm_mul_ps(X1, persp[1]);
    preX[2] = _mm_mul_ps(X2, persp[2]);
    dest[0] = _mm_add_ps(_mm_add_ps(preX[0], preX[1]), preX[2]);

    const __m128 Y0 = _mm_shuffle_ps(attribues[0], attribues[0], _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 Y1 = _mm_shuffle_ps(attribues[1], attribues[1], _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 Y2 = _mm_shuffle_ps(attribues[2], attribues[2], _MM_SHUFFLE(1, 1, 1, 1));

    __m128 preY[3];
    preY[0] = _mm_mul_ps(Y0, persp[0]);
    preY[1] = _mm_mul_ps(Y1, persp[1]);
    preY[2] = _mm_mul_ps(Y2, persp[2]);
    dest[1] = _mm_add_ps(_mm_add_ps(preY[0], preY[1]), preY[2]);

    const __m128 Z0 = _mm_shuffle_ps(attribues[0], attribues[0], _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 Z1 = _mm_shuffle_ps(attribues[1], attribues[1], _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 Z2 = _mm_shuffle_ps(attribues[2], attribues[2], _MM_SHUFFLE(2, 2, 2, 2));

    __m128 preZ[3];
    preZ[0] = _mm_mul_ps(Z0, persp[0]);
    preZ[1] = _mm_mul_ps(Z1, persp[1]);
    preZ[2] = _mm_mul_ps(Z2, persp[2]);
    dest[2] = _mm_add_ps(_mm_add_ps(preZ[0], preZ[1]), preZ[2]);

    const __m128 W0 = _mm_shuffle_ps(attribues[0], attribues[0], _MM_SHUFFLE(3, 3, 3, 3));
    const __m128 W1 = _mm_shuffle_ps(attribues[1], attribues[1], _MM_SHUFFLE(3, 3, 3, 3));
    const __m128 W2 = _mm_shuffle_ps(attribues[2], attribues[2], _MM_SHUFFLE(3, 3, 3, 3));

    __m128 preW[3];
    preW[0] = _mm_mul_ps(W0, persp[0]);
    preW[1] = _mm_mul_ps(W1, persp[1]);
    preW[2] = _mm_mul_ps(W2, persp[2]);
    dest[3] = _mm_add_ps(_mm_add_ps(preW[0], preW[1]), preW[2]);

    // Comvert from:                to:
    // dest[0] = X, X, X, X         dest[0] = X, Y, Z, W
    // dest[1] = Y, Y, Y, Y         dest[1] = X, Y, Z, W
    // dest[2] = Z, Z, Z, Z         dest[2] = X, Y, Z, W
    // dest[3] = W, W, W, W         dest[3] = X, Y, Z, W
    _MM_TRANSPOSE4_PS(dest[0], dest[1], dest[2], dest[3]);
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
static inline __m128 _Phong_Shading_Get_Colour(const __m128 weights[3], const mvec4 world_space_coords[3], const mvec4 normals[3], const Light_t *light)
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

    const mvec4 shading_colour = Light_Calculate_Shading((mvec4){.m = frag_position}, (mvec4){.m = frag_normal}, global_app.camera_position, light);

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
    const Texture_t normal_map = global_app.obj.bump;

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
    const Texture_t texture = global_app.obj.diffuse;

    ASSERT(res_u <= texture.w);
    ASSERT(res_v <= texture.h);
    ASSERT(texture.data);

    // Calculate the texture index in the texture image
    const unsigned int index = (res_u * texture.w + res_v) * texture.bpp;

    const unsigned char *pixel_data = texture.data + index;

    return _mm_div_ps(_mm_setr_ps(pixel_data[0], pixel_data[1], pixel_data[2], 255.0f),
                      _mm_set1_ps(255.0f));
}

// NOTE : Temporary
static inline float hsum_ps_sse3(const __m128 v)
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

#define COMPUTE_AREA_IN_RASTER

void Flat_Shading(const RasterData_t rd[4], const uint8_t collected_triangles_count)
{
    Texture_t tex = global_app.obj.diffuse;
    ASSERT(tex.data);

    Texture_t nrm = global_app.obj.bump;
    ASSERT(nrm.data);

    Light_t light_data = global_app.light;

    float   *pDepthBuffer  = global_renderer.z_buffer_array;
    uint8_t *pColourBuffer = global_renderer.pixels;

    const __m128i x_pixel_offset = _mm_setr_epi32(0, 1, 2, 3); // NOTE: The "r" here!! makes loading and storing colour easier
    const __m128i y_pixel_offset = _mm_setr_epi32(0, 0, 0, 0);

    /* 4 triangles, 3 vertices, 4 * 3 = 12 'x' values, we can store all this in  X_values[3]*/
    __m128i X_values[3], Y_values[3]; // ints
    __m128  Z_values[3], W_values[3]; // floats
    for (uint8_t i = 0; i < 3; i++)
    {
        /* Get 4 verticies at once */
        __m128 tri0_vert_i = rd[0].screen_space_verticies[i].m; // Get vertex i from triangle 0
        __m128 tri1_vert_i = rd[1].screen_space_verticies[i].m; // Get vertex i from triangle 1
        __m128 tri2_vert_i = rd[2].screen_space_verticies[i].m; // Get vertex i from triangle 2
        __m128 tri3_vert_i = rd[3].screen_space_verticies[i].m; // Get vertex i from triangle 3

        _MM_TRANSPOSE4_PS(tri0_vert_i, tri1_vert_i, tri2_vert_i, tri3_vert_i);

        tri0_vert_i = _mm_add_ps(tri0_vert_i, _mm_set1_ps(0.5f));
        X_values[i] = _mm_cvtps_epi32(tri0_vert_i);

        tri0_vert_i = _mm_add_ps(tri1_vert_i, _mm_set1_ps(0.5f));
        Y_values[i] = _mm_cvtps_epi32(tri1_vert_i);

        Z_values[i] = tri2_vert_i;
        W_values[i] = tri3_vert_i;
    }

    // Counter clockwise triangles.. I hope :D
    const __m128i A0 = _mm_sub_epi32(Y_values[1], Y_values[2]);
    const __m128i A1 = _mm_sub_epi32(Y_values[2], Y_values[0]);
    const __m128i A2 = _mm_sub_epi32(Y_values[0], Y_values[1]);

    const __m128i B0 = _mm_sub_epi32(X_values[2], X_values[1]);
    const __m128i B1 = _mm_sub_epi32(X_values[0], X_values[2]);
    const __m128i B2 = _mm_sub_epi32(X_values[1], X_values[0]);

    // Compute C = (xa * yb - xb * ya) for the 3 line segments that make up each triangle
    const __m128i C0 = _mm_sub_epi32(_mm_mullo_epi32(X_values[1], Y_values[2]), _mm_mullo_epi32(X_values[2], Y_values[1]));
    const __m128i C1 = _mm_sub_epi32(_mm_mullo_epi32(X_values[2], Y_values[0]), _mm_mullo_epi32(X_values[0], Y_values[2]));
    const __m128i C2 = _mm_sub_epi32(_mm_mullo_epi32(X_values[0], Y_values[1]), _mm_mullo_epi32(X_values[1], Y_values[0]));

#ifdef COMPUTE_AREA_IN_RASTER
    const __m128i triArea = _mm_sub_epi32(_mm_mullo_epi32(B2, A1), _mm_mullo_epi32(B1, A2));
    // const __m128  oneOverTriArea = _mm_rcp_ps(_mm_cvtepi32_ps(triArea));
    const __m128 oneOverTriArea = _mm_div_ps(_mm_set1_ps(1.0f), _mm_cvtepi32_ps(triArea));
#endif

    // Use bounding box traversal strategy to determine which pixels to rasterize
    const __m128i startX = _mm_and_si128(_mm_max_epi32(_mm_min_epi32(_mm_min_epi32(X_values[0], X_values[1]), X_values[2]), _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFE));
    const __m128i endX   = _mm_min_epi32(_mm_add_epi32(_mm_max_epi32(_mm_max_epi32(X_values[0], X_values[1]), X_values[2]), _mm_set1_epi32(1)), _mm_set1_epi32(global_renderer.width));

    const __m128i startY = _mm_and_si128(_mm_max_epi32(_mm_min_epi32(_mm_min_epi32(Y_values[0], Y_values[1]), Y_values[2]), _mm_set1_epi32(0)), _mm_set1_epi32(0xFFFFFFFE));
    const __m128i endY   = _mm_min_epi32(_mm_add_epi32(_mm_max_epi32(_mm_max_epi32(Y_values[0], Y_values[1]), Y_values[2]), _mm_set1_epi32(1)), _mm_set1_epi32(global_renderer.height));

    // Light position will be changed with normal mapping, we need to preserve the old value
    mvec4 world_space_light_positon = light_data.position;

    /* lane is the counter for how many triangles were loaded, if only 3 were loaded, it
        should only be 3, etc...
    */
    for (uint8_t lane = 0; lane < collected_triangles_count; lane++) // Now we have 4 triangles set up.  Rasterize them each individually.
    {
#ifdef COMPUTE_AREA_IN_RASTER
        const float area_value = oneOverTriArea.m128_f32[lane];
        // if (area_value <= 0.0f)
        //     continue;
#else
        const float area_value = vertex_data[lane].area;
#endif
        const __m128 inv_area = _mm_set1_ps(area_value);

        if (global_app.shading_mode == SHADING_WIRE_FRAME)
        {
            const float *v0 = rd[lane].screen_space_verticies[0].f;
            const float *v1 = rd[lane].screen_space_verticies[1].f;
            const float *v2 = rd[lane].screen_space_verticies[2].f;

            const SDL_Colour col = {255, 255, 255, 255};
            _draw_line((int)v0[0], (int)v0[1], (int)v1[0], (int)v1[1], &col);
            _draw_line((int)v1[0], (int)v1[1], (int)v2[0], (int)v2[1], &col);
            _draw_line((int)v2[0], (int)v2[1], (int)v0[0], (int)v0[1], &col);

#if 0 /* Drawing normals */
        const SDL_Colour col2 = {255, 000, 000, 255};
        if (mate_dot(rd.normals[0], global_app.camera_position) < 0.0f)
            Draw_Line((int)v0[0], (int)v0[1], (int)rd.endpoints[0].f[0], (int)rd.endpoints[0].f[1], &col2);

        if (mate_dot(rd.normals[1], global_app.camera_position) < 0.0f)
            Draw_Line((int)v1[0], (int)v1[1], (int)rd.endpoints[1].f[0], (int)rd.endpoints[1].f[1], &col2);

        if (mate_dot(rd.normals[2], global_app.camera_position) < 0.0f)
            Draw_Line((int)v2[0], (int)v2[1], (int)rd.endpoints[2].f[0], (int)rd.endpoints[2].f[1], &col2);
#endif
            continue;
        }

        __m128 tang_cam_pos[3]   = {0};
        __m128 tang_light_pos[3] = {0};
        __m128 tang_world_pos[3] = {0};
        if (global_app.shading_mode == SHADING_NORMAL_MAPPING)
        {
            for (size_t i = 0; i < 3; i++)
            {
                const mmat3 TBN = rd[lane].TBN[i];

                tang_light_pos[i] = mate_mat3_mulv4(TBN, world_space_light_positon).m;  // Tangent light position
                tang_cam_pos[i]   = mate_mat3_mulv4(TBN, global_app.camera_position).m; // Tangent camera position

                tang_world_pos[i] = mate_mat3_mulv4(TBN, rd[lane].world_space_verticies[i]).m;
            }
        }

        const int startXx = startX.m128i_i32[lane];
        const int endXx   = endX.m128i_i32[lane];
        const int startYy = startY.m128i_i32[lane];
        const int endYy   = endY.m128i_i32[lane];

        __m128 U[3];
        U[0] = _mm_set1_ps(rd[lane].tex_u.f[0]);
        U[1] = _mm_set1_ps(rd[lane].tex_u.f[1]);
        U[2] = _mm_set1_ps(rd[lane].tex_u.f[2]);

        __m128 V[3];
        V[0] = _mm_set1_ps(rd[lane].tex_v.f[0]);
        V[1] = _mm_set1_ps(rd[lane].tex_v.f[1]);
        V[2] = _mm_set1_ps(rd[lane].tex_v.f[2]);

        __m128 Z[3];
        Z[0] = _mm_set1_ps(Z_values[0].m128_f32[lane]);
        Z[1] = _mm_set1_ps(Z_values[1].m128_f32[lane]);
        Z[2] = _mm_set1_ps(Z_values[2].m128_f32[lane]);

        const __m128 ws_vertices[3] = {
            rd[lane].world_space_verticies[0].m,
            rd[lane].world_space_verticies[1].m,
            rd[lane].world_space_verticies[2].m,
        };

        const __m128 normals[3] = {
            rd[lane].normals[0].m,
            rd[lane].normals[1].m,
            rd[lane].normals[2].m,
        };

        // GOURAUD Shading
        __m128 gourand_colours[3];
        if (global_app.shading_mode == SHADING_GOURAUD) // We interpolate the colours in the "Vertex Shader"
        {
            mvec4 tmp_colours0 = Light_Calculate_Shading(rd[lane].world_space_verticies[0], rd[lane].normals[0], global_app.camera_position, &light_data);
            mvec4 tmp_colours1 = Light_Calculate_Shading(rd[lane].world_space_verticies[1], rd[lane].normals[1], global_app.camera_position, &light_data);
            mvec4 tmp_colours2 = Light_Calculate_Shading(rd[lane].world_space_verticies[2], rd[lane].normals[2], global_app.camera_position, &light_data);

            gourand_colours[0] = tmp_colours0.m;
            gourand_colours[1] = tmp_colours1.m;
            gourand_colours[2] = tmp_colours2.m;
        }

        __m128i a0 = _mm_set1_epi32(A0.m128i_i32[lane]);
        __m128i a1 = _mm_set1_epi32(A1.m128i_i32[lane]);
        __m128i a2 = _mm_set1_epi32(A2.m128i_i32[lane]);

        __m128i b0 = _mm_set1_epi32(B0.m128i_i32[lane]);
        __m128i b1 = _mm_set1_epi32(B1.m128i_i32[lane]);
        __m128i b2 = _mm_set1_epi32(B2.m128i_i32[lane]);

        // Add our SIMD pixel offset to our starting pixel location, so we are doing 4 pixels in the x axis
        // so we add 0, 1, 2, 3, to the starting x value, y isnt changing
        __m128i col = _mm_add_epi32(x_pixel_offset, _mm_set1_epi32(startXx));
        __m128i row = _mm_add_epi32(y_pixel_offset, _mm_set1_epi32(startYy));

        __m128i A0_start = _mm_mullo_epi32(a0, col);
        __m128i A1_start = _mm_mullo_epi32(a1, col);
        __m128i A2_start = _mm_mullo_epi32(a2, col);

        /* Step in the y direction */
        // First we must compute E at out starting pixel, this will be the minX and minY of
        // our boudning box of the traingle
        __m128i B0_start = _mm_mullo_epi32(b0, row);
        __m128i B1_start = _mm_mullo_epi32(b1, row);
        __m128i B2_start = _mm_mullo_epi32(b2, row);

        // Barycentric Setip
        // Order of triangle sides *IMPORTANT*
        // E(x, y) = a*x + b*y + c;
        // v1, v2 :  w0_row = (A12 * p.x) + (B12 * p.y) + C12;
        __m128i E0 = _mm_add_epi32(_mm_add_epi32(A0_start, B0_start), _mm_set1_epi32(C0.m128i_i32[lane]));
        __m128i E1 = _mm_add_epi32(_mm_add_epi32(A1_start, B1_start), _mm_set1_epi32(C1.m128i_i32[lane]));
        __m128i E2 = _mm_add_epi32(_mm_add_epi32(A2_start, B2_start), _mm_set1_epi32(C2.m128i_i32[lane]));

        // Since we are doing SIMD, we need to calcaulte our step amount
        // E(x+L, y) = E(x) + L dy (where dy is out a0 values)
        // B0_inc controls the step amount in the Y axis, since we are only moving 1px at a time in the y axis
        // we dont need to change the step amount
        __m128i B0_inc = b0;
        __m128i B1_inc = b1;
        __m128i B2_inc = b2;

        // A0_inc controls the step amount in the X axis, we are doing 4px at a time so multiply our dY by 4
        __m128i A0_inc = _mm_slli_epi32(a0, 2); // a0 * 4
        __m128i A1_inc = _mm_slli_epi32(a1, 2);
        __m128i A2_inc = _mm_slli_epi32(a2, 2);

        const __m128 one_over_w0 = _mm_set1_ps(rd[lane].w_values[0]);
        const __m128 one_over_w1 = _mm_set1_ps(rd[lane].w_values[1]);
        const __m128 one_over_w2 = _mm_set1_ps(rd[lane].w_values[2]);

        // Generate masks used for tie-breaking rules (not to double-shade along shared edges)
        // there is no _mm_cmpge_epi32, so use lt and swap operands
        // _mm_cmplt_epi32(bb0Inc, _mm_setzero_si128()) - becomes - _mm_cmplt_epi32(_mm_setzero_si128(), bb0Inc)
        const __m128i Edge0TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(A0_inc, _mm_setzero_si128()),
                                                   _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), B0_inc), _mm_cmpeq_epi32(A0_inc, _mm_setzero_si128())));

        const __m128i Edge1TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(A1_inc, _mm_setzero_si128()),
                                                   _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), B1_inc), _mm_cmpeq_epi32(A1_inc, _mm_setzero_si128())));

        const __m128i Edge2TieBreak = _mm_or_epi32(_mm_cmpgt_epi32(A2_inc, _mm_setzero_si128()),
                                                   _mm_and_epi32(_mm_cmplt_epi32(_mm_setzero_si128(), B2_inc), _mm_cmpeq_epi32(A2_inc, _mm_setzero_si128())));

        // Rasterize
        // Incrementally compute Fab(x, y) for all the pixels inside the bounding box formed by (startX, endX) and (startY, endY)
        for (size_t pix_y = startYy; pix_y < endYy; ++pix_y,
                    E0    = _mm_add_epi32(E0, B0_inc),
                    E1    = _mm_add_epi32(E1, B1_inc),
                    E2    = _mm_add_epi32(E2, B2_inc))
        {
            // Compute barycentric coordinates
            __m128i alpha = E0;
            __m128i betaa = E1;
            __m128i gamma = E2;

            for (size_t pix_x = startXx; pix_x < endXx; pix_x += 4,
                        alpha = _mm_add_epi32(alpha, A0_inc),
                        betaa = _mm_add_epi32(betaa, A1_inc),
                        gamma = _mm_add_epi32(gamma, A2_inc))
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

                // Early out if all of this quad's pixels are outside the triangle.
                if (_mm_test_all_zeros(mask, mask))
                    continue;

                const __m128 w0 = _mm_mul_ps(_mm_cvtepi32_ps(alpha), inv_area); // Bary A
                const __m128 w1 = _mm_mul_ps(_mm_cvtepi32_ps(betaa), inv_area); // B
                const __m128 w2 = _mm_mul_ps(_mm_cvtepi32_ps(gamma), inv_area); // C

                const __m128 w_vals[3] = {
                    _mm_mul_ps(one_over_w0, w0),
                    _mm_mul_ps(one_over_w1, w1),
                    _mm_mul_ps(one_over_w2, w2),
                };

                // Compute barycentric-interpolated depth (OpenGL Spec - Page 427)
                __m128 gl_FragCoord_z = _mm_add_ps(_mm_add_ps(_mm_mul_ps(Z[0], w0), _mm_mul_ps(Z[1], w1)), _mm_mul_ps(Z[2], w2));

                __m128 gl_FragCoord_w = _mm_add_ps(_mm_add_ps(w_vals[0], w_vals[1]), w_vals[2]);
                gl_FragCoord_w        = _mm_rcp_ps(gl_FragCoord_w);

                __m128 persp[3];
                persp[0] = _mm_mul_ps(w_vals[0], gl_FragCoord_w);
                persp[1] = _mm_mul_ps(w_vals[1], gl_FragCoord_w);
                persp[2] = _mm_mul_ps(w_vals[2], gl_FragCoord_w);

                const size_t index  = pix_y * global_renderer.width + pix_x;
                float       *pDepth = &pDepthBuffer[index];

                // DEPTH BUFFER
                const __m128 previousDepthValue           = _mm_load_ps(pDepth);
                const __m128 are_new_depths_less_than_old = _mm_cmplt_ps(gl_FragCoord_z, previousDepthValue);

                if ((uint16_t)_mm_movemask_ps(are_new_depths_less_than_old) == 0x0000)
                    continue;

                const __m128 which_depths_should_be_drawn = _mm_and_ps(are_new_depths_less_than_old, _mm_castsi128_ps(mask));
                const __m128 updated_depth_values         = _mm_blendv_ps(previousDepthValue, gl_FragCoord_z, which_depths_should_be_drawn);
                _mm_store_ps(pDepth, updated_depth_values);

                const __m128i finalMask = _mm_castps_si128(which_depths_should_be_drawn);

                // Loop over each pixel and draw
                __m128i combined_colours = {0};
                if (global_app.shading_mode == SHADING_DEPTH_BUFFER)
                {
                    continue;
                }
                else if (global_app.shading_mode == SHADING_FLAT)
                {
                    combined_colours = _mm_set1_epi32(0x00FF00FF);
                }
                else if (global_app.shading_mode == SHADING_TEXTURED || global_app.shading_mode == SHADING_NORMAL_MAPPING || global_app.shading_mode == SHADING_TEXTURED_PHONG)
                {
                    __m128 newU[3];
                    newU[0] = _mm_mul_ps(U[0], persp[0]);
                    newU[1] = _mm_mul_ps(U[1], persp[1]);
                    newU[2] = _mm_mul_ps(U[2], persp[2]);

                    __m128 newV[3];
                    newV[0] = _mm_mul_ps(V[0], persp[0]);
                    newV[1] = _mm_mul_ps(V[1], persp[1]);
                    newV[2] = _mm_mul_ps(V[2], persp[2]);

                    __m128 U_w = _mm_add_ps(_mm_add_ps(newU[0], newU[1]), newU[2]);
                    __m128 V_w = _mm_add_ps(_mm_add_ps(newV[0], newV[1]), newV[2]);

                    // clamp the vector to the range [0.0f, 1.0f]
                    U_w = _mm_max_ps(_mm_min_ps(U_w, _mm_set1_ps(1.0f)), _mm_setzero_ps());
                    V_w = _mm_max_ps(_mm_min_ps(V_w, _mm_set1_ps(1.0f)), _mm_setzero_ps());

                    U_w = _mm_mul_ps(U_w, _mm_set1_ps((float)tex.w - 1));
                    V_w = _mm_mul_ps(V_w, _mm_set1_ps((float)tex.h - 1));

                    // (U + texture.width * V) * texture.bpp
                    __m128i texture_offset = _mm_mullo_epi32(
                        _mm_set1_epi32(tex.bpp),
                        _mm_add_epi32(
                            _mm_cvtps_epi32(U_w),
                            _mm_mullo_epi32(
                                _mm_set1_epi32(tex.w),
                                _mm_cvtps_epi32(V_w))));

                    const __m128i which_tex_coords_to_get = _mm_abs_epi32(finalMask);
                    texture_offset                        = _mm_mullo_epi32(texture_offset, which_tex_coords_to_get);

                    __m128i pixel_colour[4] = {0};
                    for (int i = 0; i < 4; ++i)
                        pixel_colour[i] = _mm_loadu_si128((__m128i *)(tex.data + texture_offset.m128i_u32[i]));

                    if (global_app.shading_mode == SHADING_TEXTURED_PHONG)
                    {
                        // Interpolate pixel location
                        __m128 inter_frag_location[4] = {0};
                        _Interpolate_Something(persp, ws_vertices, inter_frag_location);

                        __m128 inter_normal[4] = {0};
                        _Interpolate_Something(persp, normals, inter_normal);

                        for (int i = 0; i < 4; ++i)
                        {
                            light_data.diffuse_colour.m = _mm_setr_ps(pixel_colour[i].m128i_u8[0] / 255.0f,
                                                                      pixel_colour[i].m128i_u8[1] / 255.0f,
                                                                      pixel_colour[i].m128i_u8[2] / 255.0f,
                                                                      1.0f);

                            mvec4 shading0 = Light_Calculate_Shading((mvec4){.m = inter_frag_location[i]},
                                                                     (mvec4){.m = inter_normal[i]},
                                                                     global_app.camera_position,
                                                                     &light_data);

                            pixel_colour[i] = _mm_cvtps_epi32(_mm_mul_ps(shading0.m, _mm_set1_ps(255.0f)));
                        }

                        const __m128i interleaved1 = _mm_packus_epi32(pixel_colour[0], pixel_colour[1]);
                        const __m128i interleaved2 = _mm_packus_epi32(pixel_colour[2], pixel_colour[3]);

                        combined_colours = _mm_packus_epi16(interleaved1, interleaved2);
                    }
                    else if (global_app.shading_mode == SHADING_NORMAL_MAPPING)
                    {
                        // 1 - get normal from normal map (in the range -1 to 1)
                        // 2 - get diffuse colour from diffuse texture
                        // 3 - interpolate the cam position, light position and the frag position in tangent space
                        // 4 - calculate shading
                        // 5 - convert back to 255 colours

                        __m128 tang_frag_pos[4] = {0};
                        _Interpolate_Something(persp, tang_world_pos, tang_frag_pos);

                        __m128 tang_frag_cam_pos[4] = {0};
                        _Interpolate_Something(persp, tang_cam_pos, tang_frag_cam_pos);

                        __m128 tang_frag_light_pos[4] = {0};
                        _Interpolate_Something(persp, tang_light_pos, tang_frag_light_pos);

                        // Could this be the same as "texture_offset"? nope, nrm map is 3bpp
                        __m128i nrm_texture_offset = _mm_mullo_epi32(
                            _mm_set1_epi32(nrm.bpp),
                            _mm_add_epi32(
                                _mm_cvtps_epi32(U_w),
                                _mm_mullo_epi32(
                                    _mm_set1_epi32(nrm.w),
                                    _mm_cvtps_epi32(V_w))));

                        for (int i = 0; i < 4; ++i) // For each pixel
                        {
                            // normal[0] = ((nrmRGB[0] / 255.0f) * 2.0f) - 1.0f;
                            unsigned char *nrm_ptr    = nrm.data + nrm_texture_offset.m128i_u32[i];
                            __m128         loaded_nrm = _mm_setr_ps(nrm_ptr[0], nrm_ptr[1], nrm_ptr[2], 0.0f);

                            loaded_nrm = _mm_sub_ps(
                                _mm_mul_ps(
                                    _mm_div_ps(loaded_nrm, _mm_set1_ps(255.0f)),
                                    _mm_set1_ps(2.0f)),
                                _mm_set1_ps(1.0f));

                            light_data.diffuse_colour.m = _mm_setr_ps(pixel_colour[i].m128i_u8[0] / 255.0f,
                                                                      pixel_colour[i].m128i_u8[1] / 255.0f,
                                                                      pixel_colour[i].m128i_u8[2] / 255.0f,
                                                                      1.0f);

                            light_data.position.m = tang_frag_light_pos[i];

                            mvec4 shading = Light_Calculate_Shading((mvec4){.m = tang_frag_pos[i]},
                                                                    (mvec4){.m = loaded_nrm},
                                                                    (mvec4){.m = tang_frag_cam_pos[i]},
                                                                    &light_data);

                            pixel_colour[i] = _mm_cvtps_epi32(_mm_mul_ps(shading.m, _mm_set1_ps(255.0f)));
                        }

                        const __m128i interleaved1 = _mm_packus_epi32(pixel_colour[0], pixel_colour[1]);
                        const __m128i interleaved2 = _mm_packus_epi32(pixel_colour[2], pixel_colour[3]);

                        combined_colours = _mm_packus_epi16(interleaved1, interleaved2);
                    }
                    else
                    {
                        const __m128i interleaved1 = _mm_unpacklo_epi32(pixel_colour[0], pixel_colour[1]);
                        const __m128i interleaved2 = _mm_unpacklo_epi32(pixel_colour[2], pixel_colour[3]);

                        // combined: [r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3, r4, g4, b4, a4]
                        combined_colours = _mm_unpacklo_epi64(interleaved1, interleaved2);
                    }
                }
                else
                {
                    __m128i pixel_colour[4] = {0};
                    if (global_app.shading_mode == SHADING_GOURAUD) // GOURAND Shading ------
                    {
                        __m128 inter_colour[4] = {0}; // this will return us  ( inter_colour[0] = R,inter_colour[1] = G, inter_colour[2] = B )
                        _Interpolate_Something(persp, gourand_colours, inter_colour);

                        // _MM_FROUND_TO_NEAREST_INT: rounding should be performed to the nearest integer value
                        // _MM_FROUND_NO_EXC: rounding should not generate any exceptions (stops exception if the input is NaN (Not a Number))
                        const __m128 cvt_colout0 = _mm_round_ps(_mm_mul_ps(inter_colour[0], _mm_set1_ps(255.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        const __m128 cvt_colout1 = _mm_round_ps(_mm_mul_ps(inter_colour[1], _mm_set1_ps(255.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        const __m128 cvt_colout2 = _mm_round_ps(_mm_mul_ps(inter_colour[2], _mm_set1_ps(255.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        const __m128 cvt_colout3 = _mm_round_ps(_mm_mul_ps(inter_colour[3], _mm_set1_ps(255.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                        pixel_colour[0] = _mm_cvtps_epi32(cvt_colout0);
                        pixel_colour[1] = _mm_cvtps_epi32(cvt_colout1);
                        pixel_colour[2] = _mm_cvtps_epi32(cvt_colout2);
                        pixel_colour[3] = _mm_cvtps_epi32(cvt_colout3);
                    }
                    else if (global_app.shading_mode == SHADING_PHONG || global_app.shading_mode == SHADING_BLIN_PHONG)
                    {
                        // Interpolate pixel location
                        __m128 inter_frag_location[4] = {0};
                        _Interpolate_Something(persp, ws_vertices, inter_frag_location);

                        __m128 inter_normal[4] = {0};
                        _Interpolate_Something(persp, normals, inter_normal);

                        mvec4 shading0 = Light_Calculate_Shading((mvec4){.m = inter_frag_location[0]}, (mvec4){.m = inter_normal[0]}, global_app.camera_position, &light_data);
                        mvec4 shading1 = Light_Calculate_Shading((mvec4){.m = inter_frag_location[1]}, (mvec4){.m = inter_normal[1]}, global_app.camera_position, &light_data);
                        mvec4 shading2 = Light_Calculate_Shading((mvec4){.m = inter_frag_location[2]}, (mvec4){.m = inter_normal[2]}, global_app.camera_position, &light_data);
                        mvec4 shading3 = Light_Calculate_Shading((mvec4){.m = inter_frag_location[3]}, (mvec4){.m = inter_normal[3]}, global_app.camera_position, &light_data);

                        pixel_colour[0] = _mm_cvtps_epi32(_mm_mul_ps(shading0.m, _mm_set1_ps(255.0f)));
                        pixel_colour[1] = _mm_cvtps_epi32(_mm_mul_ps(shading1.m, _mm_set1_ps(255.0f)));
                        pixel_colour[2] = _mm_cvtps_epi32(_mm_mul_ps(shading2.m, _mm_set1_ps(255.0f)));
                        pixel_colour[3] = _mm_cvtps_epi32(_mm_mul_ps(shading3.m, _mm_set1_ps(255.0f)));
                    }

                    // 32bit values into 16 bit values
                    const __m128i packed1 = _mm_packus_epi32(pixel_colour[0], pixel_colour[1]);
                    const __m128i packed2 = _mm_packus_epi32(pixel_colour[2], pixel_colour[3]);

                    // move the 16bit values, into the 8 bit values
                    combined_colours = _mm_packus_epi16(packed1, packed2);
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
}