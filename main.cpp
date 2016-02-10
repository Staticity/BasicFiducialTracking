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

bool find_black_quad(Mat image, vector<Point>& quad)
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

    Mat thresh;
    threshold(
        gray,
        thresh,
        100,
        255,
        THRESH_BINARY
    );

    double maxLabel;
    minMaxLoc(labels, NULL, &maxLabel);

    int labelCount = (int)(maxLabel) + 1;
    vector<int> blackCount(labelCount);
    vector<vector<Point> > points(labelCount);

    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label        = labels.at<int>(i, j);
            int thresh_value = thresh.at<uchar>(i, j);

            blackCount[label] += thresh_value == 0;  
            points[label].push_back(Point(j, i));
        }
    }

    float percent_black_acc = 0.90;
    int min_size_acc = 500;

    vector<int> candidates_s1;
    for (int i = 0; i < labelCount; ++i)
    {
        int numPixels = points[i].size();
        float percent_black = ((float)blackCount[i]) / max(1, numPixels);
        if (percent_black >= percent_black_acc && numPixels >= min_size_acc)
        {
            candidates_s1.push_back(i);
        }
    }


    vector<int> candidates_s2;
    vector<vector<Point> > polys;
    for (int i = 0; i < candidates_s1.size(); ++i)
    {
        int index = candidates_s1[i];
        vector<Point> hull;
        convexHull(Mat(points[index]), hull, true);
        approxPolyDP(hull, hull, 3, true);

        if (hull.size() == 4)
        {
            candidates_s2.push_back(index);
            polys.push_back(hull);
        }
    }

    int best_index = -1;
    int best_size = -1;
    for (int i = 0; i < candidates_s2.size(); ++i)
    {
        int index = candidates_s2[i];
        int numPixels = points[index].size();
        if (numPixels > best_size)
        {
            best_index = i;
            best_size = numPixels;
        }
    }
    
    if (best_index == -1)
        return false;

    quad = polys[best_index];

    return true;
}

void sort_quad_corners(Mat image,
                       const vector<Point>& quad,
                       vector<Point>& sorted_quad)
{

}

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    VideoCapture vc(index);
    int width = 640;
    int height = 480;
    vc.set(CV_CAP_PROP_FRAME_WIDTH, width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    if (!vc.isOpened()) return 0;

    Mat image;
    vc >> image;

    Mat drawing = Mat::zeros(image.size(), CV_8UC3);

    vector<Point> quad;
    if (1)
    {
        while (vc.isOpened())
        {
            vc >> image;
            bool found = find_black_quad(image, quad);

            Scalar color;
            if (!found)
            {
                color = Scalar(0, 0, 255);
            }
            else
            {
                color = Scalar(0, 255, 0);
            }

            if (quad.size() == 4)
            {
                drawing = Mat::zeros(image.size(), CV_8UC3);
                for (int i = 0; i <= 4; ++ i)
                {
                    int i1 = i % 4;
                    int i2 = (i + 1) % 4;
                    line(drawing, quad[i1], quad[i2], color);
                }
            }

            imshow("black_quad_outline", drawing);
            imshow("source_image", image);

            if (waitKey(20) == 27)
                break;
        }
    }
}
