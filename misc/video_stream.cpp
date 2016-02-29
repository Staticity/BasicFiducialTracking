#include <opencv.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // double thresh1;
    // std::cin >> thresh1;
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
        double t = getTickCount();
/*        cvtColor(image, image, CV_BGR2HSV);

        vector<Mat> planes;
        split(image, planes);

        // cout << planes[0] << endl;
        planes[1] = Mat::zeros(planes[0].size(), CV_8UC1);
        planes[2] = Mat::zeros(planes[0].size(), CV_8UC1);

        merge(&planes[0], 3, image);
        // cvtColor(image, image, CV_HSV2BGR);

        std::cout << (((double) getTickCount()) - t) / getTickFrequency() * 1000.0 << std::endl;
        // Canny(image, image, thresh1, thresh1 * 0.5);
*/
        imshow("test", image);
        if (waitKey(30) == 27) break;
    }
}
