#include <stdio.h>
#include <stdlib.h>

#include "ObjLoader.h"

int main()
{
    const char *filename = "../../res/Wooden Box/wooden crate.obj";

    const Mesh_Data mesh = Get_Object_Data(filename, false);

    fprintf(stderr, "vertex_count = %d\n", mesh.num_of_triangles);

    for (unsigned int i = 0; i < 24; i++)
    {
        /* code */
        fprintf(stderr, "[%d] %f\n", i, mesh.vertex_coordinates[i]);
    }

    return 0;
}
