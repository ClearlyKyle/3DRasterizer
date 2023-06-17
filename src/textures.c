#include "textures.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

inline void Texture_Print_Info(const Texture_t t)
{
    fprintf(stderr, "Texture width  : %d\n", t.w);
    fprintf(stderr, "Texture height : %d\n", t.h);
    fprintf(stderr, "Texture bbp    : %d\n", t.bpp);
}

Texture_t Texture_Load(const char *file_path, bool flip)
{
    Texture_t t = {0};

    // textures oriented tha same as you view them in paint
    stbi_set_flip_vertically_on_load(flip ? 1 : 0);

    unsigned char *data = stbi_load(file_path, &t.w, &t.h, &t.bpp, 0);
    ASSERTF(data, "Error! Loading Texture : %s\n", file_path);

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