#include <opencv.hpp>

#include <iostream>

#include "Util.hpp"
#include "CameraData.hpp"
#include "VanillaMatcher.hpp"
#include "VanillaTracker.hpp"
#include "CameraData.hpp"

using namespace std;
using namespace cv;

// class Tracker
// =============
// struct Input
// {
//     std::vector<cv::KeyPoint>  feat1;
//     std::vector<cv::KeyPoint>  feat2;
//     std::vector<cv::DMatch>    matches;
//     CameraData                 camera;
// };

// struct Output
// {        
//     cv::Mat                    fundamental;
//     cv::Mat                    essential;
//     cv::Mat                    essentialU;
//     cv::Mat                    essentialW;
//     cv::Mat                    essentialVt;
    
//     cv::Mat                    rotation;
//     cv::Mat                    translation;
//     std::vector<CloudPoint>    points;

//     float                      visible_percent;
//     float                      avg_reprojection_error;
// };

void print_usage_format()
{
    printf("[arg1]: %s\n[arg2]: %s\n[arg3]: %s\n",
        "<camera_calibration_filepath>",
        "<video_width>",
        "<video_height>");
}

int main(int argc, char** argv)
{
    int expected_argument_count = 3;
    if (argc - 1 < expected_argument_count)
    {
        print_usage_format();
        return -1;
    }

    string calibration_filepath = argv[1];
    int video_width             = atoi(argv[2]);
    int video_height            = atoi(argv[3]);

    CameraData camera(calibration_filepath);
    camera.resize(Size(video_width, video_height));

    VideoCapture vc(0);
    vc.set(CV_CAP_PROP_FRAME_WIDTH, video_width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, video_height);

    if (!vc.isOpened()) return 0;

    Mat source;
    Mat query = imread("images/Lenna.png");

    Mat                   desc1, desc2;
    std::vector<KeyPoint> feat1, feat2;
    std::vector<Point2d>   pts1,  pts2;
    std::vector<DMatch>   matches;
    while (vc.isOpened())
    {
        vc >> source;

        feat2.clear();
        desc2 = Mat();
        matches.clear();

        VanillaMatcher matcher;
        matcher.match(query, source, feat1, feat2, desc1, desc2, matches);

        Util::toPoints(feat1, pts1);
        Util::toPoints(feat2, pts2);

        vector<char> mask;
        Mat homography = findHomography(pts1, pts2, CV_RANSAC, 2, mask);

        Mat drawing;
        drawMatches(query, feat1, source, feat2, matches, drawing,
                    Scalar::all(-1), Scalar::all(-1), mask);

        imshow("drawing", drawing);

        char c = waitKey(20);
        if      (c ==  27) break;
    }
}
