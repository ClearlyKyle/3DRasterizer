#ifndef __TEST_SQAURE_H__
#define __TEST_SQAURE_H__

// Triangle 1 : lower right
// W      Z      Y      X
// 1.0f, -1.0f,  1.0f,  1.0f
// 1.0f, -1.0f, -1.0f,  1.0f
// 1.0f, -1.0f, -1.0f, -1.0f

// Triangle 1 : lower right
// W      Z      Y      X
// 1.0f, -1.0f,  1.0f, -1.0f
// 1.0f, -1.0f,  1.0f,  1.0f
// 1.0f, -1.0f, -1.0f, -1.0f

float test_square_x[] = {
    1.0f,
    1.0f,
    -1.0f,
    -1.0f,
    1.0f,
    -1.0f};

float test_square_y[] = {
    1.0f,
    -1.0f,
    -1.0f,
    1.0f,
    1.0f,
    -1.0f,
};

float test_square_z[] = {
    -1.0f,
    -1.0f,
    -1.0f,
    -1.0f,
    -1.0f,
    -1.0f};

float test_square_w[] = {
    1.0f,
    1.0f,
    1.0f,
    1.0f,
    1.0f,
    1.0f};

float test_square_tex_u[] = {
    0.0f,
    1.0f,
    0.0f,
    1.0f,
    1.0f,
    0.0f};

float test_square_tex_v[] = {
    1.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f};

#endif // __TEST_SQAURE_H__