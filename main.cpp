#include <opencv.hpp>

#include <iostream>

#include "Util.hpp"
#include "CameraData.hpp"
#include "FlowMatcher.hpp"
#include "VanillaTracker.hpp"
#include "CameraData.hpp"

using namespace std;
using namespace cv;

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

    Mat source, query;
    // Mat query = imread("images/Lenna.png");

    FlowMatcher matcher;
    VanillaTracker tracker;

    Tracker::Input in(camera);

    Mat                   desc1, desc2;
    std::vector<KeyPoint> feat1, feat2;
    std::vector<Point2d>   pts1,  pts2;
    std::vector<DMatch>  matches;
    vc >> query;

    while (vc.isOpened())
    {
        vc >> source;

        feat1.clear();
        feat2.clear();
        matches.clear();
        bool success = matcher.match(query, source, feat1, feat2, desc1, desc2, matches);

        if (success)
        {
            pts1.clear();
            pts2.clear();

            Util::toPoints(feat1, feat2, matches, pts1, pts2);

            vector<uchar> mask(pts1.size());
            Mat homography = findHomography(pts1, pts2, CV_RANSAC, 2, mask);

            Util::retain(vector<Point2d>(pts1), mask, pts1);
            Util::retain(vector<Point2d>(pts2), mask, pts2);

            for (int i = 0; i < pts1.size(); ++i)
            {
                arrowedLine(source, pts2[i], pts1[i], Scalar(0, 255, 0), 1);
            }

            in.pts1 = pts1;
            in.pts2 = pts2;

            Tracker::Output out;
            success = tracker.triangulate(in, out);
            std::cout << (success ? "true" : "false") << std::endl;
            std::cout << out.points.size() << std::endl;
            // std::cout << out.rotation << std::endl;
            // std::cout << out.translation << std::endl;

            // if (success)
            // {
            //     std::vector<Point2d> matched2d;
            //     std::vector<Point3d> matched3d;
            //     for (int i = 0; i < out.points.size(); ++i)
            //     {
            //         matched2d.push_back(pts2[out.points[i].index]);
            //         matched3d.push_back(out.points[i].pt);
            //         cout << matched3d[i].x << " " << matched3d[i].y << " " << matched3d[i].z << endl;
            //         // Scalar color = source.at<Scalar>(matched2d[i].y, matched2d[i].x);
            //         // cout << color[0] << " " << color[1] << " " << color[2] << endl;
            //     }
            //     waitKey();
            // }
        }
        // else 
        {
            imshow("drawing", source);
        }

        query = source.clone();
        char c = waitKey(20);
        if      (c ==  27) break;
    }
}
