#ifndef __TIMER_H__
#define __TIMER_H__

#include "SDL2/SDL.h"

struct timer
{
    Uint64 start;
    Uint64 elapsed;
    Uint64 perf_frequency;
    // char  *name;
};

#define TIMER_INIT(NAME) \
    struct timer(NAME) = {0}

#define TIMER_START(NAME) \
    Timer_Start(&(NAME))

#define TIMER_UPDATE(NAME) \
    Timer_Update(&(NAME))

#define TIMER_STOP(NAME) \
    Timer_Stop(&(NAME))

inline void Timer_Start(struct timer *const timer)
{
    timer->perf_frequency = SDL_GetPerformanceFrequency();
    timer->start          = SDL_GetPerformanceCounter();
    timer->elapsed        = 0;
}

inline struct timer Timer_Init_Start(void)
{
    struct timer t = {0};
    Timer_Start(&t);
    return t;
}

inline void Timer_Update(struct timer *const timer)
{
    const Uint64 new_time = SDL_GetPerformanceCounter();
    timer->elapsed        = new_time - timer->start;
    timer->start          = new_time;
}

inline void Timer_Stop(struct timer *const timer)
{
    timer->elapsed = SDL_GetPerformanceCounter() - timer->start;
}

inline double Timer_Get_Elapsed_Seconds(const struct timer *const timer)
{
    return (double)timer->elapsed / (double)timer->perf_frequency;
}

inline double Timer_Get_Elapsed_MS(const struct timer *const timer)
{
    return Timer_Get_Elapsed_Seconds(timer) * 1000.0;
}

// inline void Timer_Display(struct timer *timer)
//{
//     fprintf(stdout, "%s - elapsed seconds: %f\n", timer->name, Timer_Get_Elapsed_Seconds(timer));
// }

#endif // __TIMER_H__