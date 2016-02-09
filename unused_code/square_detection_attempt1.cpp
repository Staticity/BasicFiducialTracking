#include <cv.h>
#include <highgui.h>
#include <imgproc/imgproc.hpp>

#include <string>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

map<uchar, int> frequency_gray_in_contour(Mat img, const vector<Point>& poly)
{
    Mat gray = img;
    
    if (img.channels() != 1)
        cvtColor(gray, gray, CV_BGR2GRAY);

    map<uchar, int> frequencies;
    Rect r = boundingRect(poly);
    int x1 = (int)(r.x);
    int y1 = (int)(r.y);
    int x2 = x1 + (int)(r.width);
    int y2 = y1 + (int)(r.height);

    for (int j = x1; j <= x2; ++j)
    {
        for (int k = y1; k <= y2; ++k)
        {
            // Is this point of the bounding rect in the quad?
            if (!(pointPolygonTest(poly, Point(j, k), false) < -0.5f))
            {
                ++frequencies[img.at<uchar>(k, j)];
            }
        }
    }

    return frequencies;
}

bool find_square(Mat src, vector<vector<Point> >& contours, vector<Point>& outputSq)
{
    if (src.channels() != 1)
    {
        return false;
    }

    // Find quads -- may need revision
    vector<vector<Point> > polys;
    for (int i = 0; i < contours.size(); ++i)
    {
        vector<Point> approximatePoly;
        approxPolyDP(contours[i], approximatePoly, 3, true);

        // if (approximatePoly.size() >= 4 &&
            // approximatePoly.size() <= 6)
        {
            polys.push_back(approximatePoly);
        }
    }

    map<uchar, int> best_freq;
    int   best_index    = -1;
    int   best_size     = -1;
    float percentage    = 0.0f;
    float percent_black = 0.90f;
    int   min_total_acc = 500;

    // Find quad with at least 90% black of greatest size
    for (int i = 0; i < polys.size(); ++i)
    {
        map<uchar, int> freq = frequency_gray_in_contour(src, polys[i]);
        map<uchar, int>::iterator it;

        int total = 0;
        for (it = freq.begin(); it != freq.end(); ++it)
        {
            total += it->second;
        }

        float cur_percent = (0.0 + freq[0]) / max(total, 1);
        if (cur_percent > percent_black && total >= min_total_acc)
        {
            if (best_index == -1 || total > best_size)
            {
                best_index = i;
                best_size = total;
                percentage = cur_percent;
                best_freq = freq;
            }
        }
    }

    if (best_index == -1)
    {
        // no satisfactory quad found
        return false;
    }

    cout << best_size << " " << polys[best_index].size() << " " << percentage << " " << contourArea(polys[best_index]) << endl;
    // std::cout << best_index << " " << best_size << " " << percentage << std::endl;
    map<uchar, int>::iterator it;
    for (it = best_freq.begin(); it != best_freq.end(); ++it)
    {
        cout << "Frequency: " << (int)it->first << ":\t" << it->second << endl;
    }
    outputSq = polys[best_index];

    return true;
}

vector<vector<Point> > get_contours(Mat src, int threshold=100, bool debug_draw_contours=false)
{
    Mat img = src.clone();

    // Convert to grayscale
    if (img.channels() != 1)
    {
        cvtColor(img, img, CV_BGR2GRAY);
    }

    // Canny edge detection
    Mat canny;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    Canny(img, canny, threshold, threshold * 2, 3);
    findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    if (debug_draw_contours)
    {
        RNG rng(0);
        Mat drawing = Mat::zeros(canny.size(), CV_8UC3);
        for (int i = 0; i < contours.size(); ++i)
        {
            // approxPolyDP(contours[i], contours[i], 3, true);
            // if (contours[i].size() != 4) continue;
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            drawContours(drawing, contours, i, color, 0, 8, hierarchy, 0, Point());
        }

        namedWindow("Contours", CV_WINDOW_AUTOSIZE);
        imshow("Contours", drawing);
    }

    return contours;
}*/

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    VideoCapture vc(index);
    Mat image, gray, thresh;

    if (!vc.isOpened()) return 0;
    vc >> image;

    namedWindow("Source");
    // namedWindow("Square");
    while (vc.isOpened())
    {
        vc >> image;
        cvtColor(image, gray, CV_BGR2GRAY);
        adaptiveThreshold(
            gray,
            thresh,
            255,
            ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY,
            7,
            10);

        vector<vector<Point> > contours;
        contours = get_contours(thresh, 100, true);

        threshold(gray, thresh, 50, 255, THRESH_BINARY);

        vector<Point> sq;
        if (find_square(thresh, contours, sq))
        {
            Mat drawing = Mat::zeros(thresh.size(), CV_8UC3);
            vector<vector<Point> > squares;
            squares.push_back(sq);
            drawContours(drawing, squares, 0, Scalar(255, 0, 0), 1, 8);
            imshow("Square", drawing);
        }

        imshow("Source", thresh);

        if (waitKey(30) == 27)
            break;
    }
}
