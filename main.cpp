#include <opencv.hpp>
//#include <opencv/highgui/highgui.hpp>
//#include <imgproc.hpp>

#include <string>
#include <iostream>
#include <map>
#include <ctime>

using namespace cv;
using namespace std;

void assignLabels(Mat image, Mat& drawing)
{
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

    int erosion_size = 2;
    int square_size  = erosion_size * 2 + 1;
    Mat element = getStructuringElement(
        MORPH_RECT,
        Size(square_size, square_size),
        Point(erosion_size, erosion_size)
    );

    erode(adapt, adapt, element);

    Mat labels;
    connectedComponents(adapt, labels, 4);

    double maxLabel;
    minMaxLoc(labels, NULL, &maxLabel);

    RNG rng(1234);
    vector<Vec3b> colors;
    for (int i = 0; i <= maxLabel; ++i)
    {
        colors.push_back(Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }

    cout << colors.size() << endl;
    
    drawing = Mat(image.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; ++i)
        for (int j = 0; j < labels.cols; ++j)
            drawing.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
}

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    VideoCapture vc(index);

    if (!vc.isOpened()) return 0;

    Mat image, drawing;
    vc >> image;

    if (1)
    {
        while (vc.isOpened())
        {
            vc >> image;
            assignLabels(image, drawing);

            imshow("Labels", drawing);

            waitKey(20);
//            if (waitKey(20) == 27)
//                break;
        }
    }
}
