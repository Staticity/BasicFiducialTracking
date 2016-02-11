#include <opencv.hpp>
#include <string>

using namespace cv;

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

        imshow("test", image);
        if (waitKey(30) == 27) break;
    }
}
