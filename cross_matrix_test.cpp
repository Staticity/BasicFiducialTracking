#include <opencv.hpp>
#include <assert.h>

#include <cstdio>

using namespace std;
using namespace cv;

int main()
{
    Vec<double, 3> x1(5, 132, 4321);
    Vec<double, 3> x2(4.0, 5.0, 6.0);

    Vec<double, 3> actual_cross = x1.cross(x2);
    assert(actual_cross[0] = x1[1] * x2[2] - x1[2] * x2[1]);
    assert(actual_cross[1] = x1[0] * x2[2] - x1[2] * x2[0]);
    assert(actual_cross[2] = x1[0] * x2[1] - x1[1] * x2[0]);

    double mat_data[3][3] =
    {
        {     0, -x1[2],   x1[1]},
        {-x1[2],      0,   x1[0]},
        {-x1[1],  x1[0],       0},
    };


    Mat mat_cross(3, 3, CV_64F, mat_data);

    cout << Mat(actual_cross) - mat_cross * Mat(x2) << endl;
}