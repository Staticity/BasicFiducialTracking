#include <cv.h>
#include <highgui.h>
#include <imgproc/imgproc.hpp>

#include <string>
#include <iostream>
#include <map>
#include <ctime>

#include "disjoint_set.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    VideoCapture vc(index);

    if (!vc.isOpened()) return 0;

    Mat image;
    vc >> image;

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    
    Mat adapt;
    adaptiveThreshold(
        gray,
        adapt,
        255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY,
        7,
        10
    );

//    Mat labels;
//    connectedComponents(adapt, labels, 4);

//    imshow("Labels", labels);

    if (0)
    {
        namedWindow("Source");
        while (vc.isOpened())
        {
            vc >> image;
            cvtColor(image, gray, CV_BGR2GRAY);

            imshow("Source", image);

            if (waitKey(20) == 27)
                break;
        }
    }
}
