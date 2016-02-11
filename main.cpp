#include <opencv.hpp>
#include <string>

using namespace cv;

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    VideoCapture vc(index);
    Mat image;

    if (!vc.isOpened()) return 0;
    vc >> image;

    namedWindow("test");
    while (vc.isOpened())
    {
        vc >> image;

        imshow("test", image);
        if (waitKey(30) == 27) break;
    }
}
