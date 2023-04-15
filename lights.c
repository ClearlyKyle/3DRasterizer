#include "lights.h"

#include "Renderer.h"

static __m128 Reflect_m128(const __m128 I, const __m128 N)
{
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/reflect.xhtml
    __m128 dotProduct = _mm_dp_ps(I, N, 0x7f);                                 // Compute dot product of I and N
    __m128 scaledN    = _mm_mul_ps(N, _mm_mul_ps(dotProduct, _mm_set1_ps(2))); // Scale N by 2 * dotProduct
    return _mm_sub_ps(I, scaledN);                                             // Subtract scaledN from I to get reflected vector
}

/**
 * Calculates the diffuse component of the lighting equation for a given light direction, and surface normal
 *
 * @param L             The direction of the light source
 * @param N             The surface normal at the point on the surface where the light is being calculated
 *
 * @return An __m128 vector containing the diffuse amount, with values between 0 and 1.0f, the first component
 *          is set to 1.0f to represent the alpha value, and the remaining three components are set to the diffuse amount
 */
static __m128 Calculate_Diffuse_Amount(const __m128 L, const __m128 N)
{
    const float dot_product = Calculate_Dot_Product_SIMD(N, L);

    const float diffuse_amount = (const float)fmax(dot_product, 0.0);

    return _mm_set_ps(1.0f, diffuse_amount, diffuse_amount, diffuse_amount);
}

/**
 * Calculates the specular amount for a given light and view direction, surface normal, and shininess factor.
 *
 * @param L             Light direction
 * @param E             View direction
 * @param N             Surface normal
 * @param shininess     A material property that determines the size and concentration of the specular highlight on a surface
 *                      1.0f (low shininess, wide specular highlight) to 1000.0f or higher (high shininess, sharp specular highlight)
 *
 * @return An __m128 vector containing the specular power, with values between 0 and 1.0f,
 *          the first component set to 1 (representing the alpha value)
 */
static __m128 Calculate_Specular_Amount(const __m128 L, const __m128 E, const __m128 N, const float shininess)
{
    float specular_power = 0.0f;

    if (global_app.shading_mode == BLIN_PHONG)
    {
        // Calculate the Halfway vector (H) between the light source direction (L) and the view direction (E)
        const __m128 H = Normalize_m128(_mm_add_ps(L, E));

        // Calculate the specular component using the dot product of the surface normal (N) and the halfway vector (H)
        const float dot_product = Calculate_Dot_Product_SIMD(N, H);
        specular_power          = powf(fmaxf(dot_product, 0.0), shininess);
    }
    else
    {
        // Calculate R - the reflection vector
        const __m128 R = Reflect_m128(L, N);

        const float dot_product = Calculate_Dot_Product_SIMD(R, E);
        specular_power          = powf(fmaxf(dot_product, 0.0), shininess);
    }

    return _mm_set_ps(1.0f, specular_power, specular_power, specular_power);
}

/**
 * Calculates the shading for a given point on a surface based on the lighting information.
 *
 * @param position         The position of the point on the surface (GOURAND uses world position of the triangle verticies, while PHONG uses the pixel position)
 * @param normal           The normal vector of the surface at the point
 * @param camera_position  The position of the camera viewing the surface
 * @param light            A pointer to a Light struct containing information about the light source
 *
 * @return An __m128 vector containing the RGB values of the shading at the point, with values between 0 and 255.
 */
__m128 Light_Calculate_Shading(const __m128 position, const __m128 normal, const Light *light)
{
    // Normalise the Noraml
    const __m128 N = Normalize_m128(normal);

    // Calculate L - direction to the light source
    const __m128 L = Normalize_m128(_mm_sub_ps(light->position, position));

    // Calculate E - view direction
    const __m128 E = Normalize_m128(_mm_sub_ps(global_app.camera_position, position));

    // Calculate Ambient Term:
    const __m128 Iamb = light->ambient_colour;

    // Calculate Diffuse Term:
    const __m128 diffuse_amount = Calculate_Diffuse_Amount(L, N);
    const __m128 Idiff          = _mm_mul_ps(light->diffuse_colour, diffuse_amount); // Might need to set the Alpha here

    // Calculate Specular Term:
    const float  shininess = 32.0f;
    const __m128 Ispec     = Calculate_Specular_Amount(L, E, N, shininess);

    // const __m128 colour = _mm_add_ps(_mm_add_ps(Iamb, Idiff), Ispec);
    const __m128 colour = _mm_mul_ps(Clamp_m128(_mm_add_ps(_mm_add_ps(Iamb, Idiff), Ispec), 0.0f, 1.0f), _mm_set1_ps(255.0f));
    return colour;
}
