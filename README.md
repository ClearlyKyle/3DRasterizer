3D rasterizer written in C that utilizes SIMD instructions, OpenMP. Capable of loading objects from OBJ files and displaying them using various shading techniques. The supported shading techniques are as follows:

- SHADING_WIRE_FRAME: Renders the objects using wireframe rendering, which displays only the edges of the polygons.
- SHADING_FLAT: Applies flat shading to the objects, where each polygon is filled with a single color.
- SHADING_GOURAUD: Implements Gouraud shading, which interpolates vertex colors across the polygon faces to achieve smoother shading.
- SHADING_PHONG: Utilizes Phong shading, which calculates per-pixel lighting using interpolated vertex normals.
- SHADING_BLIN_PHONG: Applies Blinn-Phong shading, a modification of Phong shading that uses the halfway vector between the light direction and view direction to calculate lighting.
- SHADING_TEXTURED: Enables texture mapping on the objects, allowing textures to be applied to the polygons.
- SHADING_TEXTURED_PHONG: Combines texture mapping with Phong shading, providing both per-pixel lighting and texture information.
- SHADING_NORMAL_MAPPING: Implements normal mapping, a technique that perturbs the surface normals of the polygons using a texture, giving the illusion of greater surface detail.
- SHADING_DEPTH_BUFFER: Utilizes a depth buffer to perform hidden surface removal, ensuring that only the closest visible polygons are rendered.

Features
Loads objects from OBJ files, allowing you to import 3D models into the rasterizer.
Supports a variety of shading techniques for rendering objects.
Utilizes SIMD instructions to optimize performance by performing parallel computations.
Takes advantage of OpenMP to parallelize rendering tasks across multiple threads.


Prerequisites
C compiler compatible with MSVC.
SIMD-enabled processor to take advantage of the SIMD optimizations.
OpenMP library for parallelization support.

Getting Started
Clone the repository or download the source code.