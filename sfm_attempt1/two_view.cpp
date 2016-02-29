#include <opencv.hpp>

#include <iostream>
#include <fstream>

#include "Util.hpp"
#include "Camera.hpp"
#include "HybridMatcher.hpp"
#include "VanillaTracker.hpp"

using namespace std;
using namespace cv;

string header_format = "\
ply\n\
format ascii 1.0\n\
element vertex %d\n\
property float x\n\
property float y\n\
property float z\n\
property uchar r\n\
property uchar g\n\
property uchar b\n\
end_header\n";

void print_usage_format()
{
    printf("[arg1]: %s\n[arg2]: %s\n[arg3]: %s\n[arg4]L %s\n",
        "<camera_calibration_filepath>",
        "<image_1>",
        "<image_2>",
        "<save_cloud_filename>");
    // printf("[arg1]: %s\n[arg2]: %s\n[arg3]: %s\n",
    //     "<camera_calibration_filepath>",
    //     "<save_cloud_filename>",
    //     "<camera_index>");
}

int main(int argc, char** argv)
{
    int expected_argument_count = 4;
    
    if (argc - 1 < expected_argument_count)
    {
        print_usage_format();
        return -1;
    }

    string calibration_filepath = argv[1];
    string image1_filepath      = argv[2];
    string image2_filepath      = argv[3];
    string cloud_save_file      = argv[4];

    Mat im1 = imread(image1_filepath);
    Mat im2 = imread(image2_filepath);
    
    /*
    int expected_argument_count = 3;
    if (argc - 1 < expected_argument_count)
    {
        print_usage_format();
        return -1;
    }

    string calibration_filepath = argv[1];
    string cloud_save_file      = argv[2];
    int camera_index            = atoi(argv[3]);

    Mat im, im1, im2;
    VideoCapture vc(camera_index);

    int index = 1;
    while (vc.isOpened())
    {
        vc >> im;
        imshow("Source", im);
        char c = waitKey(20);

        if (c == ' ')
        {
            if (index == 1)
            {
                im1 = im.clone();
                ++index;
            }
            else
            {
                im2 = im.clone();
                break;
            }
        }
        else if (c == 27) return -1;
    }
    destroyWindow("Source");
    */

    if (im1.size() != im2.size())
    {
        printf("Expecting images to be of the same size\n");
        return -1;
    }

    Camera camera(calibration_filepath);
    camera.resize(im1.size());

    HybridMatcher matcher;
    VanillaTracker tracker;

    std::vector<KeyPoint> feat1, feat2;
    Mat desc1, desc2;
    std::vector<DMatch>  matches;
    bool success = matcher.match(im1, im2, feat1, feat2, desc1, desc2, matches);

    if (!success)
    {
        printf("Couldn't match features.\n");
        return -1;
    }

    std::vector<Point2d> pts1,  pts2;
    Util::toPoints(feat1, feat2, matches, pts1, pts2);
    assert(pts1.size() == pts2.size());

    Mat feature_matches = im2.clone();

    for (int i = 0; i < pts1.size(); ++i)
    {
        arrowedLine(feature_matches, pts2[i], pts1[i], Scalar(0, 255, 0), 1);
    }
    imshow("Feature Matches", feature_matches);

    Tracker::Output out;
    Tracker::Input in(camera);
    in.pts1 = pts1;
    in.pts2 = pts2;

    success = tracker.triangulate(in, out);

    if (!success)
    {
        printf("Couldn't triangulate.\n");
        waitKey();
        return -1;
    }

    double minV = out.points[0].pt.z;
    double maxV = minV;

    for (int i = 1; i < out.points.size(); ++i)
    {
        double z = out.points[i].pt.z;
        minV = (z < minV) ? z : minV;
        maxV = (z > maxV) ? z : maxV;
    }

    Mat depth_map = Mat::zeros(im1.size(), CV_8UC3);
    for (int i = 0; i < out.points.size() && !Util::feq(maxV - minV, 0.0); ++i)
    {
        int j = out.points[i].index;
        double z = out.points[i].pt.z;
        double s = 1.0 - ((z - minV) / (maxV - minV));
        assert(s <= 1.0 && s >= 0.0);
        
        Scalar r = Scalar(0, 0, 255);
        Scalar g = Scalar(0, 255, 0);
        Scalar c = r * s + g * (1 - s);
        // Scalar c = Scalar(255 * (1 - s), 255, 255);

        circle(depth_map, pts2[j], 1, c, CV_FILLED);
    }
    imshow("Depth", depth_map);

    char buffer[1000];
    sprintf(buffer, header_format.c_str(), out.points.size());
    string header(buffer);

    ofstream myfile;
    myfile.open(cloud_save_file);
    myfile << header;
    for (int i = 0; i < out.points.size(); ++i)
    {
        Point3d cloudPoint = out.points[i].pt;
        Point2d imagePoint = pts2[out.points[i].index];
        Vec3b   color      = im1.at<Vec3b>(imagePoint);
        myfile << cloudPoint.x << " " << cloudPoint.y << " " << cloudPoint.z << " " << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << '\n';
    }
    myfile.close();
    waitKey();
}
