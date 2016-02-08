#include <cv.h>
#include <highgui.h>

using namespace cv;

int main()
{
    VideoCapture vc(0);
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
