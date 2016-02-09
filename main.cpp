#include <opencv.hpp>
//#include <opencv/highgui/highgui.hpp>
//#include <imgproc.hpp>

#include <string>
#include <iostream>
#include <map>
#include <ctime>
#include <assert.h>

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

    Mat thresh;
    threshold(
        gray,
        thresh,
        80,
        1,
        THRESH_BINARY
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

    int labelCount = (int)(maxLabel) + 1;
    vector<int> blackCount(labelCount);
    vector<int>  numPixels(labelCount);

    // for (int i = 0; i < labelCount; ++i)
        // assert(blackCount[i] == 0 && numPixels[i] == 0);

    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label        = labels.at<int>(i, j);
            int thresh_value = thresh.at<uchar>(i, j);
            // assert(thresh_value == 0 || thresh_value == 1);

            blackCount[label] += thresh_value == 0;  
            ++numPixels[label];
        }
    }


    vector<int> candidates;
    // vector<bool> is_candidate(labelCount);
    float percent_black_acc = 0.90;
    int min_size_acc = 1000;
    for (int i = 0; i < labelCount; ++i)
    {
        float percent_black = ((float)blackCount[i]) / max(1, numPixels[i]);
        if (percent_black >= percent_black_acc && numPixels[i] >= min_size_acc)
        {
            candidates.push_back(i);
            // is_candidate[i] = true;
        }
    }

    int best_index = 0;
    int best_size = 0;
    for (int i = 0; i < candidates.size(); ++i)
    {
        if (numPixels[candidates[i]] > best_size)
        {
            best_index = candidates[i];
            best_size = numPixels[i];
        }
    }

    RNG rng(1234);
    vector<Vec3b> colors(labelCount);
    for (int i = 0; i < labelCount; ++i)
    {
        colors[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    drawing = Mat::zeros(image.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; ++i)
        for (int j = 0; j < labels.cols; ++j)
            if (labels.at<int>(i, j) == best_index)
                drawing.at<Vec3b>(i, j) = Vec3b(0, 255, 0);

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

            imshow("Source", image);
            imshow("Labels", drawing);

            waitKey(20);
//            if (waitKey(20) == 27)
//                break;
        }
    }
}
