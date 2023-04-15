#include "textures.h"

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

Texture Texture_Load(const char *file_path)
{
    Texture t = {0};

    // textures oriented tha same as you view them in paint
    stbi_set_flip_vertically_on_load(1);

    unsigned char *data = stbi_load(file_path, &t.w, &t.h, &t.bpp, 0);
    if (!data)
    {
        fprintf(stderr, "Loading image : %s\n", stbi_failure_reason());
        return t;
    }

    t.data = data;

    return t;
}