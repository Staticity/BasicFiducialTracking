#include <opencv.hpp>
//#include <opencv/highgui/highgui.hpp>
//#include <imgproc.hpp>

#include <string>
#include <iostream>
#include <map>
#include <cmath>
#include <ctime>
#include <utility>
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

void sort_quad_corners(Mat& image,
                       const vector<Point>& quad,
                       vector<Point>& sorted_quad)
{
    assert(quad.size() == 4);
    assert(&quad != &sorted_quad);

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    Mat thresh;
    threshold(
        gray,
        thresh,
        155,
        255,
        THRESH_BINARY
    );

    Rect r = boundingRect(quad);
    Point2f white_centroid = Point(0, 0);
    int num_white_points = 0;
    for (int i = 0; i < r.height; ++i)
    {
        for (int j = 0; j < r.width; ++j)
        {
            int x = j + r.x;
            int y = i + r.y;

            // if (x, y) is in the quad
            if (pointPolygonTest(quad, Point(x, y), false) > -0.5)
            {
                if (thresh.at<uchar>(y, x) != 0)
                {
                    white_centroid.x += x;
                    white_centroid.y += y;
                    ++num_white_points;
                }
            }
        }
    }

    // center of white pixels
    white_centroid /= max(1, num_white_points);

    circle(image, white_centroid, 5, Scalar(255, 0, 255));

    // find quad corner closes to white centroid
    int white_corner_index = 0;
    float white_sq_distance = -1.0f;
    for (int i = 0; i < 4; ++i)
    {
        Point2f dv = white_centroid - Point2f(quad[i]);
        float sq_distance = dv.x * dv.x + dv.y * dv.y;
        
        if (sq_distance <= white_sq_distance ||
            white_sq_distance < 0.0f)
        {
            white_corner_index = i;
            white_sq_distance = sq_distance;
        }
    }

    Point2f quad_centroid = (quad[0] + quad[1] + quad[2] + quad[3]) / 4.0f;
    vector<pair<double, int> > point_angle_tuples(4);

    circle(image, quad_centroid, 5, Scalar(255, 255, 0));

    // sort points in clockwise order
    Point2f white_vector = Point2f(quad[white_corner_index]) - quad_centroid;
    for (int i = 0; i < 4; ++i)
    {
        Point2f corner_vector = Point2f(quad[i]) - quad_centroid;
        double dot = white_vector.ddot(corner_vector);
        double det = white_vector.cross(corner_vector);
        double ang = atan2(det, dot);
        while (ang < 0.0)
            ang += 360.0;
        point_angle_tuples[i] = make_pair(ang, i);
    }

    sort(point_angle_tuples.begin(), point_angle_tuples.end());

    sorted_quad = vector<Point>(4);
    for (int i = 0; i < 4; ++i)
    {
        int sorted_index = point_angle_tuples[i].second;
        sorted_quad[i] = quad[sorted_index];
    }
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
                color = Scalar(128, 128, 128);
            }

            if (quad.size() == 4)
            {
                if (found)
                {
                    vector<Point> sorted_quad;
                    sort_quad_corners(image, quad, sorted_quad);
                    quad = sorted_quad;
                }

                for (int i = 0; i < 4; ++ i)
                {
                    int i1 = i % 4;
                    int i2 = (i + 1) % 4;
                    line(image, quad[i1], quad[i2], color, 2);
                }

                Scalar colors[4] = 
                {
                    Scalar(0, 0, 255),
                    Scalar(0, 255, 0),
                    Scalar(255, 0, 0),
                    Scalar(255, 255, 255)
                };

                for (int i = 0; i < 4; ++i)
                {
                    circle(image, quad[i], 7, colors[i], 2);
                }
            }

            // imshow("black_quad_outline", drawing);
            imshow("outlined_source", image);

            if (waitKey(20) == 27)
                break;
        }
    }
}
