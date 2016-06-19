#include <opencv.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    const int width = 1280;
    const int height = 720;
    VideoCapture vc(index);

    if (!vc.isOpened()) return 0;
    
    vc.set(CV_CAP_PROP_FRAME_WIDTH, width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    Mat image;
    vc >> image;

    namedWindow("test");
    while (vc.isOpened())
    {
        vc >> image;
        cvtColor(image, image, CV_BGR2GRAY);
        Canny(image, image, 100, 50, 3);
        imshow("test", image);
        if (waitKey(30) == 27) break;
    }
}
