#ifndef __TEXTURES_H__
#define __TEXTURES_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Texture_s
{
    int            w, h, bpp;
    unsigned char *data;
} Texture_t;

Texture_t Texture_Load(const char *file_path, bool alpha);

inline void Texture_Print_Info(const Texture_t t)
{
    fprintf(stderr, "Texture width  : %d\n", t.w);
    fprintf(stderr, "Texture height : %d\n", t.h);
    fprintf(stderr, "Texture bbp    : %d\n", t.bpp);
}

#endif // __TEXTURES_H__