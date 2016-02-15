#include <opencv.hpp>

#include <iostream>

#include "Util.hpp"
#include "CameraData.hpp"
#include "HybridMatcher.hpp"
#include "VanillaTracker.hpp"

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

    HybridMatcher matcher;
    VanillaTracker tracker;

    Tracker::Input in(camera);

    Mat                   desc1, desc2;
    std::vector<KeyPoint> feat1, feat2;
    std::vector<Point2d>   pts1,  pts2;
    std::vector<DMatch>  matches;
    vc >> query;

    // int total = 0;
    // int good = 0;
    while (vc.isOpened())
    {
        vc >> source;

        feat1.clear();
        feat2.clear();
        desc1 = Mat();
        desc2 = Mat();
        matches.clear();
        bool success = matcher.match(query, source, feat1, feat2, desc1, desc2, matches);
        query = source.clone();

        // ++total;
        if (success)
        {
            Mat drawing = Mat::zeros(source.size(), CV_8UC3);
            pts1.clear();
            pts2.clear();

            Util::toPoints(feat1, feat2, matches, pts1, pts2);
            assert(pts1.size() == pts2.size());

            vector<uchar> mask(pts1.size());
            Mat homography = findHomography(pts1, pts2, CV_RANSAC, 2, mask);

            vector<Point2d> good_pts1;
            vector<Point2d> good_pts2;
            Util::retain(pts1, mask, good_pts1);
            Util::retain(pts2, mask, good_pts2);

            for (int i = 0; i < good_pts1.size(); ++i)
            {
                arrowedLine(source, good_pts2[i], good_pts1[i], Scalar(0, 255, 0), 1);
            }

            in.pts1 = good_pts1;
            in.pts2 = good_pts2;

            Tracker::Output out;
            // success = tracker.triangulate(in, out);

            // if (out.points.size() > 0)
            // {
            //     // ++good;
            //     // cout << good << "/" << total << endl;
            //     // printf("Tracking was a %s\n", (success ? "SUCCESS" : "FAILURE"));
            //     double minV = out.points[0].pt.z;
            //     double maxV = minV;

            //     int num_bad = minV > 1.0;
            //     for (int i = 1; i < out.points.size(); ++i)
            //     {
            //         double z = out.points[i].pt.z;
            //         minV = (z < minV) ? z : minV;
            //         maxV = (z > maxV) ? z : maxV;
            //         num_bad += z > 1.0;
            //     }

            //     for (int i = 0; i < out.points.size() && !Util::feq(maxV - minV, 0.0); ++i)
            //     {
            //         int j = out.points[i].index;
            //         double z = out.points[i].pt.z;
            //         z = max(minV, min(z, maxV));
            //         double s = 1.0 - ( (z - minV) / (maxV - minV));
            //         assert(s <= 1.0 && s >= 0.0);
            //         Scalar r = Scalar(0, 0, 255);
            //         Scalar g = Scalar(0, 255, 0);
            //         Scalar c = r * s + g * (1 - s);
            //         // Scalar c = Scalar(255 * (1 - s), 255, 255);
            //         circle(drawing, good_pts2[j], 1, c, CV_FILLED);
            //     }
                // cvtColor(drawing, drawing, CV_HSV2BGR);

                // printf("Num in cloud: %lu/%lu\nVisibility: %f%%\nAverage error: %f\n", out.points.size(), pts1.size(), out.visible_percent * 100.0, out.avg_reprojection_error);
                // printf("Range of z: [%f, %f] with %d/%lu considered bad\n\n", minV, maxV, num_bad, out.points.size());
                // cout << "Rotation:\n" << endl;
                // cout << out.rotation << endl << endl;
                // cout << "Translation:" << out.translation << endl;
                // cout << endl;
                // imshow("drawing", drawing);
                // if (maxV > 1.0)
                // {
                    // cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                    // waitKey();
                // }
            // }

            // imshow("drawing", drawing);
        }
        // else 
        {
            imshow("source", source);
        }

        char c = waitKey(20);
        if      (c ==  27) break;
    }
}
