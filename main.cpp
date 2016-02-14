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
    Tracker::Output out;

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
                arrowedLine(source, pts2[i], pts1[i], Scalar(255, 0, 0), 1);
            }

            in.pts1 = pts1;
            in.pts2 = pts2;
            success = tracker.triangulate(in, out);
            std::cout << (success ? "true" : "false") << std::endl;

            // mask.clear();
            // homography = findHomography(pts1, pts2, CV_RANSAC, 1, mask);


            // Mat points = (Mat_<double>(4, 3) <<          0,          0, 1,
            //                                     query.cols,          0, 1,
            //                                     query.cols, query.rows, 1,
            //                                              0, query.rows, 1);

            // // H * dest = src, rows are points
            // Mat warped = (homography * points.t()).t();
            // assert(warped.type() == CV_64F);

            // vector<Point2d> quad;
            // const Vec3d* pts = warped.ptr<Vec3d>();
            // for (int i = 0; i < 4; ++i)
            // {
            //     double wx = pts[i][0];
            //     double wy = pts[i][1];
            //     double w  = pts[i][2];

            //     assert(w != 0.0);

            //     double x = wx / w;
            //     double y = wy / w;
            //     quad.push_back(Point2d(x, y));
            // }

            // for (int i = 0; i < 4; ++i)
            // {
            //     int j = (i + 1) % 4;
            //     line(source, quad[i], quad[j], Scalar(0, 255, 0), 3);
            // }

            // Mat drawing;
            // drawMatches(query, feat1, source, feat2, matches, drawing,
                        // Scalar::all(-1), Scalar::all(-1), mask);
            // imshow("drawing", drawing);
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
