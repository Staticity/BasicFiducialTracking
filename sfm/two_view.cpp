#include <opencv.hpp>

#include "Util.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "MultiView.hpp"

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc != 3 + 1)
    {
        cout << " <calibration_filepath>";
        cout << " <image_1_filepath>";
        cout << " <image_2_filepath>";
        cout << endl;
        return -1;
    }

    string calibration_filepath = argv[1];
    string image_1_filepath     = argv[2];
    string image_2_filepath     = argv[3];

    // Find calibration file and load images
    Camera camera(calibration_filepath);
    Mat im1 = imread(image_1_filepath);
    Mat im2 = imread(image_2_filepath);

    // Images must be of the same size for correspondences
    assert(im1.size() == im2.size());

    // Resize the camera matrix to the image sizes taken
    camera.resize(im1.size());   
    
    // Find the point matches between the two frames
    vector<KeyPoint> feat1, feat2;
    Mat desc1, desc2;
    vector<DMatch> matches;
    Features::findMatches(im1, im2, feat1, feat2, desc1, desc2, matches);

    std::vector<Point2d> pts1, pts2;
    for (int i = 0; i < matches.size(); ++i)
    {
        pts1.push_back(feat1[matches[i].queryIdx].pt);
        pts2.push_back(feat2[matches[i].trainIdx].pt);
    }

    Mat F, E;
    vector<uchar> inliers;
    MultiView::fundamental(pts1, pts2, F, inliers);
    MultiView::essential(F, camera, camera, E);

    vector<cv::Mat> rotations, translations;
    MultiView::get_rotation_and_translation(E, rotations, translations);

    int n = rotations.size();
    std::vector<std::vector<Point3d> > clouds(n);
    std::vector<double> projection_error(n), in_front_percent(n);
    for (int i = 0; i < n; ++i)
    {
        Mat rotation = rotations[i];
        Mat translation = translations[i];

        Mat no_rotation = Mat::eye(3, 3, CV_64F);
        Mat no_translation = Mat::zeros(3, 1, CV_64F);
        
        MultiView::triangulate(pts1, pts2, rotation, translation, clouds[i]);

        vector<Point2d> projected_pts1, projected_pts2;
        MultiView::project(clouds[i], no_rotation, no_translation, projected_pts1);
        MultiView::project(clouds[i], rotation, translation, projected_pts2);


    }
}
