#include "lights.h"

#include "Renderer.h"

/**
 * Calculates the diffuse component of the lighting equation for a given light direction, and surface normal
 *
 * @param L             The direction of the light source
 * @param N             The surface normal at the point on the surface where the light is being calculated
 *
 * @return The diffuse amount, with values between 0 and 1.0f, the first component
 */
static inline float Calculate_Diffuse_Amount(const mvec4 L, const mvec4 N)
{
    const float dot_product = mate_dot(N, L);

    const float diffuse_amount = (const float)fmax(dot_product, 0.0);

    return diffuse_amount;
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
 * @return An float containing the specular power, with values between 0 and 1.0f
 */
static float Calculate_Specular_Amount(const mvec4 L, const mvec4 E, const mvec4 N, const float shininess)
{
    float specular_power = 0.0f;
    float dot_product    = 0.0f;

    if (global_app.shading_mode == BLIN_PHONG || global_app.shading_mode == NORMAL_MAPPING)
    {
        // Calculate the Halfway vector (H) between the light source direction (L) and the view direction (E)
        const mvec4 H = mate_norm(mate_vec4_add(L, E));

        // Calculate the specular component using the dot product of the surface normal (N) and the halfway vector (H)
        dot_product = mate_dot(N, H);
    }
    else // PHONG
    {
        // Calculate R - the reflection vector
        const mvec4 R = mate_norm(mate_reflect(mate_negate(L), N));

        dot_product = mate_dot(E, R);
    }

    specular_power = powf(fmaxf(dot_product, 0.0f), shininess);
    return specular_power;
}

/**
 * Calculates the shading for a given point on a surface based on the lighting information.
 * We pass the camera and light position seperatly due to Normal Mapping changing these values
 *
 * @param position         The position of the point on the surface (GOURAND uses world position of the triangle verticies, while PHONG uses the pixel position)
 * @param normal           The normal vector of the surface at the point
 * @param camera_position  The position of the camera viewing the surface
 * @param light            A pointer to a Light struct containing information about the light source
 *
 * @return An __m128 vector containing the RGB values of the shading at the point, with values between 0 and 255.
 */
mvec4 Light_Calculate_Shading(const mvec4 position, const mvec4 normal, const mvec4 camera_position, const mvec4 light_position, const Light *light)
{
    // Normalise the Noraml
    // const __m128 N = normal;
    const mvec4 N = mate_norm(normal);
    //
    // Calculate L - direction to the light source
    // const __m128 L = _mm_sub_ps(light_position, position);
    const mvec4 L = mate_norm(mate_vec4_sub(light_position, position));

    // Calculate E - view direction
    // const __m128 E = _mm_sub_ps(camera_position, position);
    const mvec4 E = mate_norm(mate_vec4_sub(camera_position, position));

    // Calculate Ambient Term:
    const mvec4 Iamb = mate_vec4_mul(light->diffuse_colour, light->ambient_amount);

    // Calculate Diffuse Term:
    const float diffuse_amount = Calculate_Diffuse_Amount(L, N);
    const mvec4 Idiff          = mate_vec4_scale(light->diffuse_colour, diffuse_amount); // Might need to set the Alpha here

    // Calculate Specular Term:
    const float shininess = 32.0f;
    const float specular  = Calculate_Specular_Amount(L, E, N, shininess);
    const mvec4 Ispec     = mate_vec4_scale(light->specular_amount, specular);

    const mvec4 lighting_amount = mate_vec4_clamp(
        mate_vec4_add(
            mate_vec4_add(Iamb, Idiff),
            Ispec),
        0.0f, 1.0f);

    return lighting_amount;
}
