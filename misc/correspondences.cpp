#include <opencv.hpp>
#include <xfeatures2d.hpp>

#include <cstdio>
#include <iostream>

#include "../Camera.hpp"

using namespace std;
using namespace cv;

void print_usage_format();

bool find_correspondences(const Mat& im1, const Mat& im2,
                          vector<DMatch>& matches);

bool find_correspondences(const Mat& im1, const Mat& im2,
                          vector<KeyPoint>& feat1,
                          vector<KeyPoint>& feat2,
                          Mat& descriptor1,
                          Mat& descriptor2,
                          vector<DMatch>& matches);

void print_usage_format()
{
    printf("[arg1]: %s\n[arg2]: %s\n[arg3]: %s\n",
        "<camera_calibration_filepath>",
        "<video_width>",
        "<video_height>");
}

bool find_correspondences(const Mat& im1, const Mat& im2,
                          vector<DMatch>& matches)
{
    vector<KeyPoint> feat1, feat2;
    Mat descriptor1, descriptor2;
    return find_correspondences(
        im1,
        im2,
        feat1,
        feat2,
        descriptor1,
        descriptor2,
        matches);
}

bool find_correspondences(const Mat& im1, const Mat& im2,
                          vector<KeyPoint>& feat1,
                          vector<KeyPoint>& feat2,
                          Mat& descriptor1,
                          Mat& descriptor2,
                          vector<DMatch>& matches)
{
    bool do_feat1 = feat1.empty();
    bool do_feat2 = feat2.empty();
    bool do_desc1 = descriptor1.empty();
    bool do_desc2 = descriptor2.empty();

    Mat gray1, gray2;
    if ((do_feat1 || do_desc1) && im1.channels() != 1)
        cvtColor(im1, gray1, CV_BGR2GRAY);
    else
        gray1 = im1;

    if ((do_feat2 || do_desc2) && im2.channels() != 1)
        cvtColor(im2, gray2, CV_BGR2GRAY);
    else
        gray2 = im2;

    if (do_feat1 || do_feat2)
    {
        double t = getTickCount();
        Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(400);
        // Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
        // Ptr<FeatureDetector> detector = FastFeatureDetector::create(20);
        // Ptr<FeatureDetector> detector = BRISK::create();

        if (do_feat1) detector->detect(im1, feat1);
        if (do_feat2) detector->detect(im2, feat2);

        if (feat1.empty() || feat2.empty())
            return false;
        t = getTickCount() - t;
        t /= getTickFrequency();
        // printf("Features: %0.4f\n", t);
    }

    if (do_desc1 || do_desc2)
    {
        double t = getTickCount();
        Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();
        // Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
        // Ptr<DescriptorExtractor> extractor = ORB::create();
        // Ptr<DescriptorExtractor> extractor = xfeatures2d::BriefDescriptorExtractor::create(32, true);

        if (do_desc1) extractor->compute(im1, feat1, descriptor1);
        if (do_desc2) extractor->compute(im2, feat2, descriptor2);

        if (descriptor1.empty() || descriptor2.empty())
            return false;
        t = getTickCount() - t;
        t /= getTickFrequency();
        printf("Descriptors: %0.4f\n", t);
    }

    double t = getTickCount();
    // Change distance based on descriptor
    BFMatcher matcher = BFMatcher::BFMatcher(NORM_L2, true);
    matcher.match(descriptor1, descriptor2, matches);

    // for (int i = 0; i < cross_checked_matches.size(); ++i)
    // {
        // if (!cross_checked_matches[i].empty())
            // matches.push_back(cross_checked_matches[i][0]);
    // }
    t = getTickCount() - t;
    t /= getTickFrequency();
    printf("Matches (%ld): %0.4f\n", matches.size(), t);

    return !matches.empty();
}

// static void on_mouse(int event, int x, int y, int, void* param)
// {
//     MouseParams* mp = (MouseParams*)(param);
//     Point2d      pt = Point2d(x, y);

//     int num_points = mp->points.size();
//     if (num_points >= 4) return;

//     if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)
//     {
//         mp->points.push_back(pt);
//         return;
//     }

//     Mat drawing = mp->image.clone();
//     for (int i = 0; i < num_points; ++i)
//     {
//         circle(drawing, pt, 5, Scalar(0, 0, 0), 2);

//         int j = (i + 1) % num_points;
//         line(drawing, mp->points[i], mp->points[j], Scalar(255, 255, 255));
//     }

//     imshow(mp->window_name, drawing);
// }

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


    Camera cam(calibration_filepath);
    cam.resize(Size(video_width, video_height));

    VideoCapture vc(0);
    vc.set(CV_CAP_PROP_FRAME_WIDTH, video_width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, video_height);

    if (!vc.isOpened()) return 0;

    string window_name = "source";
    namedWindow(window_name);

    Mat query = imread("images/Lenna.png");//Mat();
    Mat image = Mat();
    vector<KeyPoint> query_feat, im_feat;
    Mat              query_desc, im_desc;

    while (vc.isOpened())
    {
        vc >> image;

        bool success = false;
        if (!query.empty())
        {
            im_feat.clear();
            im_desc = Mat();

            double time_spent = getTickCount();
            // Find 2D correspondences
            std::vector<DMatch> matches;
            success = find_correspondences(query, image,
                                           query_feat, im_feat,
                                           query_desc, im_desc,
                                           matches);

            time_spent = (getTickCount() - time_spent) / getTickFrequency();
            printf("%0.4f\n", time_spent);
            if (success)
            {
                vector<Point2d> query_pts, im_pts;
                for (int i = 0; i < matches.size(); ++i)
                {
                    query_pts.push_back(Point2d(query_feat[matches[i].queryIdx].pt));
                    im_pts.push_back(Point2d(im_feat[matches[i].trainIdx].pt));
                }

                // Find homography between correspondences
                vector<char> mask;
                Mat homography = findHomography(query_pts, im_pts, CV_RANSAC, 2, mask);

                int count = 0;
                for (int i = 0; i < mask.size(); ++i) count += (mask[i] != 0);

                success = count >= 8;

                if (success)
                {
                    Mat points = (Mat_<double>(4, 3) <<          0,          0, 1,
                                                        query.cols,          0, 1,
                                                        query.cols, query.rows, 1,
                                                                 0, query.rows, 1);

                    // H * dest = src, rows are points
                    Mat warped = (homography * points.t()).t();
                    assert(warped.type() == CV_64F);

                    vector<Point2d> quad;
                    const Vec3d* pts = warped.ptr<Vec3d>();
                    for (int i = 0; i < 4; ++i)
                    {
                        double wx = pts[i][0];
                        double wy = pts[i][1];
                        double w  = pts[i][2];

                        assert(w != 0.0);

                        double x = wx / w;
                        double y = wy / w;
                        quad.push_back(Point2d(x, y));
                    }

                    for (int i = 0; i < 4; ++i)
                    {
                        int j = (i + 1) % 4;
                        line(image, quad[i], quad[j], Scalar(0, 255, 0), 3);
                    }

                    Mat drawing;
                    drawMatches(query, query_feat, image, im_feat, matches, drawing,
                                Scalar::all(-1), Scalar::all(-1),
                                mask, DrawMatchesFlags::DEFAULT);

                    imshow(window_name, drawing);
                }
            }
        }

        if (!success)
        {
            imshow(window_name, image);
        }

        char key = waitKey(20);
        if (key == ' ')
        {
            vc >> query;
            query_feat.clear();
            query_desc = Mat();
        }
        else if (key == 'r') query = Mat();
        else if (key == 'q') break;
    }

    return 0;
}
