#include <stdio.h>
#include <stdlib.h>

#include "ObjLoader.h"

int main()
{
    // Load_Object("../../obj/Yellow_Plane/Yellow_Plane_3.obj");
    // ObjectData *mesh = Load_Object("../../obj/basic_cube.obj");
    ObjectData *mesh = Load_Object("../../obj/axis.obj");
    // Load_Object("../../obj/teapot.obj");

    fprintf(stderr, "vertex_count = %d\n", mesh->vertex_count);

    return 0;
}
