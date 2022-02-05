#ifndef __OBJLOADER_H__
#define __OBJLOADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "vector.h"

#define DEBUG_PRINT 0
#define debug_pr(fmt, ...)                     \
    do                                         \
    {                                          \
        if (DEBUG_PRINT)                       \
            fprintf(stderr, fmt, __VA_ARGS__); \
    } while (0)

typedef struct _vert_data
{
    Triangle *mesh;
    unsigned int vertex_count;
} ObjectData;

typedef struct _vert_data_contiguous
{
    float *mesh;
    unsigned int vertex_count;
} ObjectDataC;

ObjectDataC *Load_Object_Contiguous(const char *filename)
{
    FILE *fp;
    errno_t err;

    if ((err = fopen_s(&fp, filename, "r")) != 0)
    {
        fprintf_s(stderr, "cannot open file '%s'\n", filename);
        return NULL;
    }

    // get size of file
    fseek(fp, 0, SEEK_END);
    const size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char line[256];
    unsigned int f_count = 0;
    unsigned int line_count = 0;

    // count number of faces
    while (fgets(line, sizeof(line), fp))
    {
        line_count += 1;
        if (line[0] == 'f')
        {
            f_count += 1;
        }
    }
    // printf("line count = %d\n", line_count);
    fseek(fp, 0, SEEK_SET);

    ObjectDataC *result = (ObjectDataC *)malloc(sizeof(ObjectDataC));
    char **buff = (char **)malloc(sizeof(char *) * (line_count + 1));
    float *obj_mesh = (float *)malloc(sizeof(float) * f_count * 4);

    unsigned int f_index = 0;
    unsigned int count = 0;
    while (fgets(line, sizeof(line), fp))
    {
        if (line[0] == '#')
        {
            continue;
        }
        if (line[0] == ' ')
        {
            // offset_count += 1;
            continue;
        }
        if (line[0] == 'v')
        {
            buff[count++] = (char *)_strdup(line);
            continue;
        }

        if (line[0] == 'f')
        {
            // printf("\noffset = %d\n", offset_count);
            int index[3];
            sscanf_s(line, "f %d %d %d", &index[0], &index[1], &index[2]);
            debug_pr("index = {%d, %d, %d}\n", index[0], index[1], index[2]);

            for (int i = 0; i < 3; i++)
            {
                float tmp[4] = {1.0f};

                sscanf_s(buff[index[i] - 1], "v %f %f %f", &tmp[0], &tmp[1], &tmp[2]);

                debug_pr("tmp = {%f, %f, %f, %f}\n", tmp[0], tmp[1], tmp[2], tmp[3]);

                const int triangle_index = f_index * 4;
                obj_mesh[triangle_index + 0] = tmp[0];
                obj_mesh[triangle_index + 1] = tmp[1];
                obj_mesh[triangle_index + 2] = tmp[2];
                obj_mesh[triangle_index + 3] = tmp[3];
            }
            f_index++;
        }
    }
    fseek(fp, 0, SEEK_SET);

    debug_pr("Size = %lld\n", size);
    debug_pr("faces count = %d\n", f_count);

    result->vertex_count = f_count;
    result->mesh = obj_mesh;
    fclose(fp);

    return result;
}

ObjectData *Load_Object(const char *filename)
{
    FILE *fp;
    errno_t err;

    ObjectData *result = (ObjectData *)malloc(sizeof(ObjectData));

    if ((err = fopen_s(&fp, filename, "r")) != 0)
    {
        fprintf_s(stderr, "cannot open file '%s'\n", filename);
        return NULL;
    }

    // get size of file
    fseek(fp, 0, SEEK_END);
    const size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char line[256];
    unsigned int v_count = 0;
    unsigned int f_count = 0;
    unsigned int line_count = 0;

    while (fgets(line, sizeof(line), fp))
    {
        line_count += 1;
        if (line[0] == 'f')
        {
            f_count += 1;
        }
    }
    // printf("line count = %d\n", line_count);
    fseek(fp, 0, SEEK_SET);

    char **buff = (char **)malloc(sizeof(char *) * (line_count + 1));

    Triangle *obj_mesh = (Triangle *)malloc(sizeof(Triangle) * f_count);

    unsigned int f_index = 0;
    unsigned int count = 0;
    while (fgets(line, sizeof(line), fp))
    {
        if (line[0] == '#' || line[0] == ' ')
        {
            continue;
        }
        if (line[0] == 'v')
        {
            buff[count++] = (char *)_strdup(line);
            continue;
        }

        if (line[0] == 'f')
        {
            int index[3];
            sscanf_s(line, "f %d %d %d", &index[0], &index[1], &index[2]);
            debug_pr("index = {%d, %d, %d}\n", index[0], index[1], index[2]);

            Triangle t;
            for (int i = 0; i < 3; i++)
            {
                vec3 tmp = {0.0f};
                tmp.w = 1.0f;
                debug_pr("%s", buff[index[i] - 1]);

                sscanf_s(buff[index[i] - 1], "v %f %f %f", &tmp.x, &tmp.y, &tmp.z);

                debug_pr("vec3 tmp = {%f, %f, %f}\n", tmp.x, tmp.y, tmp.z);

                t.vec[i] = tmp;
            }
            obj_mesh[f_index] = t;
            f_index++;
        }
    }
    fseek(fp, 0, SEEK_SET);

    debug_pr("First Triangle :\n");
    debug_pr("v = {%f, %f, %f}\n", obj_mesh[0].vec[0].x, obj_mesh[0].vec[0].y, obj_mesh[0].vec[0].z);
    debug_pr("v = {%f, %f, %f}\n", obj_mesh[0].vec[1].x, obj_mesh[0].vec[1].y, obj_mesh[0].vec[1].z);
    debug_pr("v = {%f, %f, %f}\n", obj_mesh[0].vec[2].x, obj_mesh[0].vec[2].y, obj_mesh[0].vec[2].z);

    debug_pr("Size = %lld\n", size);
    debug_pr("v count = %d\n", v_count);
    debug_pr("f count = %d\n", f_count);

    result->vertex_count = f_count;
    result->mesh = obj_mesh;

    fclose(fp);

    return result;
}

#endif // __OBJLOADER_H__