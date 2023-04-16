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

    SDL_Surface *window_surface = SDL_GetWindowSurface(global_renderer.window);

    // Get window data
    global_renderer.surface           = window_surface;
    global_renderer.fmt               = window_surface->format;
    global_renderer.pixels            = window_surface->pixels;
    global_renderer.height            = window_surface->h;
    global_renderer.width             = window_surface->w;
    global_renderer.screen_num_pixels = window_surface->h * window_surface->w;

    // Allocate z buffer
    float *z_buff = (float *)_aligned_malloc(sizeof(float) * global_renderer.screen_num_pixels, 16);
    if (!z_buff)
    {
        fprintf(stderr, "Error with : (float *)_aligned_malloc(sizeof(float) * global_renderer.screen_num_pixels, 16);\n");
        exit(2);
    }

    global_renderer.z_buffer_array  = z_buff;
    global_renderer.max_depth_value = 10.0f;

    global_renderer.running = true;
}

void Renderer_Destroy(void)
{
    if (global_renderer.window)
    {
        SDL_DestroyWindow(global_renderer.window);
        global_renderer.window = NULL;
    }

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
    // if (x < 0 || y < 0 || x >= global_renderer.width || y >= global_renderer.height)
    //     return;

    const int index = y * global_renderer.width + x;

    // if (index > global_renderer.screen_num_pixels)
    //     return;

    global_renderer.pixels[index] = (Uint32)((alpha << 24) + (red << 16) + (green << 8) + (blue << 0));
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

void Draw_Triangle_Outline(const __m128 *verticies, const SDL_Colour col)
{
    float vert1[4];
    _mm_store_ps(vert1, verticies[0]);
    float vert2[4];
    _mm_store_ps(vert2, verticies[1]);
    float vert3[4];
    _mm_store_ps(vert3, verticies[2]);

    Draw_Line((int)vert1[0], (int)vert1[1], (int)vert2[0], (int)vert2[1], &col);
    Draw_Line((int)vert2[0], (int)vert2[1], (int)vert3[0], (int)vert3[1], &col);
    Draw_Line((int)vert3[0], (int)vert3[1], (int)vert1[0], (int)vert1[1], &col);
}

void Draw_Depth_Buffer(void)
{
    const __m128 max_depth = _mm_set1_ps(global_renderer.max_depth_value);
    const __m128 value_255 = _mm_set1_ps(255.0f);

    float *pDepthBuffer = global_renderer.z_buffer_array;

    int rowIdx = 0;
    for (int y = 0; y < global_renderer.height; y += 2, rowIdx += 2 * global_renderer.width)
    {
        int index = rowIdx;
        for (int x = 0; x < global_renderer.width; x += 2, index += 4)
        {
            __m128 depthvalues = _mm_load_ps(&pDepthBuffer[index]);
            depthvalues        = _mm_div_ps(depthvalues, max_depth);
            // depthvalues = _mm_min_ps(_mm_set1_ps(1.0f), depthvalues);
            depthvalues = _mm_mul_ps(depthvalues, value_255);

            float shading[4];
            _mm_store_ps(shading, depthvalues);

            Draw_Pixel_RGBA(x + 0, y + 0, (uint8_t)shading[3], (uint8_t)shading[3], (uint8_t)shading[3], 255);
            Draw_Pixel_RGBA(x + 1, y + 0, (uint8_t)shading[2], (uint8_t)shading[2], (uint8_t)shading[2], 255);
            Draw_Pixel_RGBA(x + 0, y + 1, (uint8_t)shading[1], (uint8_t)shading[1], (uint8_t)shading[1], 255);
            Draw_Pixel_RGBA(x + 1, y + 1, (uint8_t)shading[0], (uint8_t)shading[0], (uint8_t)shading[0], 255);
        }
    }
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
static inline __m128 _Phong_Shading_Get_Colour(const __m128 weights[3], const __m128 world_space_coords[3], const __m128 normals[3], const Light *light)
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

    // We should combine the lighting colour value and the interpolated vertex colours here...

    const __m128 shading_colour = Light_Calculate_Shading(frag_position, frag_normal, global_app.camera_position, light->position, light);

    return shading_colour;
}

static inline __m128 _Get_Normal_From_Normal_Map(const int res_v, const int res_u)
{
    const Texture normal_map = global_app.nrm;

    // Calculate the texture index in the normal map
    const unsigned int index = (res_u * normal_map.w + res_v) * normal_map.bpp;

    const unsigned char *normal_colour_data = normal_map.data + index;

    // Convert the RGB 0-255 values, to coordinates
    const float r = ((normal_colour_data[0] / 255.0f) * 2.0f) - 1.0f; // x
    const float g = ((normal_colour_data[1] / 255.0f) * 2.0f) - 1.0f; // y
    const float b = ((normal_colour_data[2] / 255.0f) * 2.0f) - 1.0f; // z

    __m128 frag_normal = _mm_set_ps(0.0f, b, g, r);
    frag_normal        = Normalize_m128(frag_normal);

    return frag_normal;
}

static inline __m128 _Get_Colour_From_Diffuse_Texture(const int res_v, const int res_u)
{
    const Texture texture = global_app.tex;

    // Calculate the texture index in the texture image
    const unsigned int index = (res_u * texture.w + res_v) * texture.bpp;

    const unsigned char *pixel_data = texture.data + index;

    return _mm_set_ps(255.0f, pixel_data[2], pixel_data[1], pixel_data[0]);
}

void Textured_Shading(const __m128 *screen_space_verticies, const __m128 *world_space_verticies, const float *w_values, const __m128 *normal_values,
                      const __m128 texture_u, const __m128 texture_v, const Mat3x3 TBN, const Light *light)
{
    // Unpack Vertex data
    const __m128 v0 = screen_space_verticies[2];
    const __m128 v1 = screen_space_verticies[1];
    const __m128 v2 = screen_space_verticies[0];

    __m128 world_v0 = world_space_verticies[2];
    __m128 world_v1 = world_space_verticies[1];
    __m128 world_v2 = world_space_verticies[0];

    __m128 colours[3]      = {0};
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
       */
        // Camera and light doesnt move, so we could move this out of here...
        // light_position  = Matrix_Multiply_Vector_SIMD(TBN.elements, light->position);            // Tangent light position
        // camera_position = Matrix_Multiply_Vector_SIMD(TBN.elements, global_app.camera_position); // Tangent camera position

        // world_v0 = Matrix_Multiply_Vector_SIMD(TBN.elements, world_v0);
        // world_v1 = Matrix_Multiply_Vector_SIMD(TBN.elements, world_v1);
        // world_v2 = Matrix_Multiply_Vector_SIMD(TBN.elements, world_v2);

        light_position  = Mat3x3_mul_m128(TBN, light->position);            // Tangent light position
        camera_position = Mat3x3_mul_m128(TBN, global_app.camera_position); // Tangent camera position

        world_v0 = Mat3x3_mul_m128(TBN, world_v0);
        world_v1 = Mat3x3_mul_m128(TBN, world_v1);
        world_v2 = Mat3x3_mul_m128(TBN, world_v2);
    }

    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

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
    float *pDepthBuffer = global_renderer.z_buffer_array;
    int    rowIdx       = (aabb.minY * global_renderer.width + 2 * aabb.minX);

    const __m128 one_over_w1 = _mm_set1_ps(w_values[2]);
    const __m128 one_over_w2 = _mm_set1_ps(w_values[1]);
    const __m128 one_over_w3 = _mm_set1_ps(w_values[0]);

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += 2 * global_renderer.width,
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

            // Precalulate uv constants
            const __m128 depth_w = _mm_mul_ps(depth, _mm_set1_ps((float)global_app.tex.w - 1.0f));
            const __m128 depth_h = _mm_mul_ps(depth, _mm_set1_ps((float)global_app.tex.h - 1.0f));

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

                    // Calculate lighting
                    const __m128 lighting_contribution = Light_Calculate_Shading(frag_position, frag_normal, camera_position, light_position, light);

                    // Combine the lighting and texture colour together
                    const __m128 diff_colour = _Get_Colour_From_Diffuse_Texture(res_v, res_u);

                    // frag_colour = _mm_mul_ps(_mm_set1_ps(1.0f), lighting_contribution);
                    frag_colour = _mm_mul_ps(diff_colour, lighting_contribution);
                }

                const uint8_t red = (uint8_t)frag_colour.m128_f32[0];
                const uint8_t gre = (uint8_t)frag_colour.m128_f32[1];
                const uint8_t blu = (uint8_t)frag_colour.m128_f32[2];
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

void Flat_Shading(const __m128 *screen_space_verticies, const __m128 *world_space_verticies, const float *w_values, const __m128 *normal_values, const Light *light)
{
    // used when checking if w0,w1,w2 is greater than 0;
    const __m128i fxptZero = _mm_setzero_si128();

    // Unpack Vertex data
    const __m128 v0 = screen_space_verticies[2];
    const __m128 v1 = screen_space_verticies[1];
    const __m128 v2 = screen_space_verticies[0];

    // Gourand Shading
    __m128 colours[3];
    if (global_app.shading_mode == GOURAND) // We interpolate the colours in the "Vertex Shader"
    {
        colours[0] = Light_Calculate_Shading(world_space_verticies[0], normal_values[0], global_app.camera_position, light->position, light);
        colours[1] = Light_Calculate_Shading(world_space_verticies[1], normal_values[1], global_app.camera_position, light->position, light);
        colours[2] = Light_Calculate_Shading(world_space_verticies[2], normal_values[2], global_app.camera_position, light->position, light);

        // We should combine the lighting colour value and the interpolated vertex colours here...
    }
    else if (global_app.shading_mode == FLAT)
    {
        colours[0] = _mm_set_ps(255.f, 0.0f, 0.0f, 128.0f);
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
    float *pDepthBuffer = global_renderer.z_buffer_array;
    int    rowIdx       = (aabb.minY * global_renderer.width + 2 * aabb.minX);

    const __m128 one_over_w1 = _mm_set1_ps(w_values[0]);
    const __m128 one_over_w2 = _mm_set1_ps(w_values[1]);
    const __m128 one_over_w3 = _mm_set1_ps(w_values[2]);

    // Rasterize
    for (int y = aabb.minY; y < aabb.maxY; y += 2,
             rowIdx += 2 * global_renderer.width,
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

                __m128 frag_colour = {0};
                if (global_app.shading_mode == FLAT)
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
                    if (global_app.shading_mode == GOURAND)
                    {
                        frag_colour = _Gourand_Shading_Get_Colour(weights, colours);
                    }
                    else if (global_app.shading_mode == PHONG || global_app.shading_mode == BLIN_PHONG)
                    {
                        frag_colour = _Phong_Shading_Get_Colour(weights, world_space_verticies, normal_values, light);
                    }
                }

                frag_colour = _mm_mul_ps(frag_colour, _mm_set1_ps(255.0f));

                const uint8_t red = (uint8_t)frag_colour.m128_f32[0];
                const uint8_t gre = (uint8_t)frag_colour.m128_f32[1];
                const uint8_t blu = (uint8_t)frag_colour.m128_f32[2];
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