#include <opencv.hpp>

#include "VanillaTracker.hpp"
#include "Camera.hpp"

using namespace std;
using namespace cv;

// Generate points randomly in a 10x10x10 cube
// centered at the origin
vector<Point3d> generatePoints(int n)
{
    RNG rng;
    vector<Point3d> points;

    while (n--)
    {
        double x = (rng.uniform(0, 1000) / 100.0) - 5.0;
        double y = (rng.uniform(0, 1000) / 100.0) - 5.0;
        double z = (rng.uniform(0, 1000) / 100.0) - 5.0;

        points.push_back(Point3d(x, y, z));
    }
}

// Generate poses around a circle
vector<Mat> generatePoses(int n)
{
    vector<Mat> poses;

    const double radius = 30.0;
    double theta = 0.0;
    Point3 up(0, 0, 1);
    Point3 target(0, 0, 0);

    for (int i = 0; i < n; ++i, theta += 2 * M_PI / n)
    {
        Point3 position = Point3(radius * cos(theta), radius * sin(theta), 0.0);
        SimpleCamera camera = SimpleCamera::Lookat(position, target, up);
        poses.push_back(camera.pose());
    }

    return poses;
}

int main(int argc, char** argv)
{

}