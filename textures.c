#include "textures.h"

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

Texture_t Texture_Load(const char *file_path, bool alpha, bool flip)
{
    UTILS_UNUSED(alpha);

    Texture_t t = {0};

    // textures oriented tha same as you view them in paint
    stbi_set_flip_vertically_on_load(flip ? 1 : 0);

    // int format = alpha ? STBI_rgb_alpha : STBI_rgb;

    unsigned char *data = stbi_load(file_path, &t.w, &t.h, &t.bpp, 0);
    if (!data)
    {
        fprintf(stderr, "Loading image : %s\n", stbi_failure_reason());
        ASSERT(data);
        return t;
    }

    t.data = data;

    return t;
}

void Texture_Destroy(Texture_t *t)
{
    if (t->data)
    {
        free(t->data);
        t->data = NULL;
    }
    *t = (Texture_t){0};
}