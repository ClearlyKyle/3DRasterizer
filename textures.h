#ifndef __TEXTURES_H__
#define __TEXTURES_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utils.h"

typedef struct Texture
{
    int            w, h, bpp;
    unsigned char *data;
} Texture_t;

void      Texture_Print_Info(const Texture_t t);
Texture_t Texture_Load(const char *file_path, bool flip);
void      Texture_Destroy(Texture_t *t);

#endif // __TEXTURES_H__