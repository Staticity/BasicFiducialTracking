#include <opencv.hpp>

#include <stdio.h>

#define PROFILE(function_name, ...)                             \
{                                                               \
    long long start = cv::getTickCount();                       \
    function_name(__VA_ARGS__);                                 \
    long long end = cv::getTickCount();                         \
    double time_spent = (end - start) / cv::getTickFrequency(); \
    printf("%s\t%s(%d)\t%s(%d)\t%f seconds\n",                  \
        __DATE__,                                               \
        __FILE__,                                               \
        __LINE__,                                               \
        #function_name,                                         \
        __VA_ARGS__,                                            \
        time_spent);                                            \
}

int run(int x)
{
    if (x == 0)
        return 1;
    return run(x - 1) + run(x - 1);
}

int main()
{
    for (int i = 0; i < 25; ++i)
        PROFILE(run, i);
}
