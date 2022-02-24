#include <stdio.h>
#include <stdlib.h>

#include "ObjLoader.h"

int main()
{
    // const char *filename = "../../res/Wooden Box/wooden crate.obj";
    //  const char *filename = "../../res/Crate/Crate1.obj";
    // const char *filename = "../../res/Sphere/sphere.obj";
    const char *filename = "../../res/Sphere/sphere_normals.obj";

    Mesh_Data *mesh;
    Get_Object_Data(filename, true, &mesh);

    fprintf(stderr, "vertex_count = %d\n", mesh->num_of_triangles);

    fprintf(stderr, "\n");
    fprintf(stderr, "UV COORDINATES\n");
    for (unsigned int i = 0; i < 5; i++)
    {
        /* code */
        fprintf(stderr, "[%d] {%f, %f} {%f, %f} {%f, %f}\n", i,
                mesh->uv_coordinates[i * 6 + 0], mesh->uv_coordinates[i * 6 + 1],
                mesh->uv_coordinates[i * 6 + 2], mesh->uv_coordinates[i * 6 + 3],
                mesh->uv_coordinates[i * 6 + 4], mesh->uv_coordinates[i * 6 + 5]);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "VERTEX COORDINATES\n");
    for (unsigned int i = 0; i < 5; i++)
    {
        /* code */
        fprintf(stderr, "[%d] {%f, %f, %f, %f} {%f, %f, %f, %f} {%f, %f, %f, %f}\n", i,
                mesh->vertex_coordinates[i * 12 + 0], mesh->vertex_coordinates[i * 12 + 1], mesh->vertex_coordinates[i * 12 + 2], mesh->vertex_coordinates[i * 12 + 3],
                mesh->vertex_coordinates[i * 12 + 4], mesh->vertex_coordinates[i * 12 + 5], mesh->vertex_coordinates[i * 12 + 6], mesh->vertex_coordinates[i * 12 + 7],
                mesh->vertex_coordinates[i * 12 + 8], mesh->vertex_coordinates[i * 12 + 9], mesh->vertex_coordinates[i * 12 + 10], mesh->vertex_coordinates[i * 12 + 11]);
    }

    fprintf(stderr, "NORMAL COORDINATES\n");
    for (unsigned int i = 0; i < 5; i++)
    {
        /* code */
        fprintf(stderr, "[%d] {%f, %f, %f, %f} {%f, %f, %f, %f} {%f, %f, %f, %f}\n", i,
                mesh->normal_coordinates[i * 12 + 0], mesh->normal_coordinates[i * 12 + 1], mesh->normal_coordinates[i * 12 + 2], mesh->normal_coordinates[i * 12 + 3],
                mesh->normal_coordinates[i * 12 + 4], mesh->normal_coordinates[i * 12 + 5], mesh->normal_coordinates[i * 12 + 6], mesh->normal_coordinates[i * 12 + 7],
                mesh->normal_coordinates[i * 12 + 8], mesh->normal_coordinates[i * 12 + 9], mesh->normal_coordinates[i * 12 + 10], mesh->normal_coordinates[i * 12 + 11]);
    }

    Free_Mesh(&mesh);

    return 0;
}
