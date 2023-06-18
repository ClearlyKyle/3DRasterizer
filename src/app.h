#ifndef __APP_H__
#define __APP_H__

#include "lights.h"
#include "obj.h"

typedef struct AppState_s
{
    ObjectData_t obj;

    Light_t light;
    mvec4   camera_position;

    Shading_Mode shading_mode;
} AppState;

extern AppState global_app;

#endif // __APP_H__