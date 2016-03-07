#include <opencv.hpp>

#include "Util.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "MultiView.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>

using namespace std;
using namespace cv;

template <typename T>
float distancePointLine(const cv::Point_<T> point, const cv::Vec<T,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::abs(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

template <typename T2>
void drawEpipolarLines(const std::string& title, const cv::Mat& F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point_<T2> > points1,
                const std::vector<cv::Point_<T2> > points2,
                const float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  /*
   * Allow color drawing
   */
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<T2,3> > epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
  }
  cv::imshow(title, outImg);
  cv::waitKey();
}

void save_ply(
    const Mat& image,
    const vector<Point3d>& points,
    const vector<Point2d>& img_points,
    const string& filename)
{
    string header = "ply\
        \nformat ascii 1.0\
        \nelement vertex " + to_string(points.size()) + "\
        \nproperty float x\
        \nproperty float y\
        \nproperty float z\
        \nproperty uchar red\
        \nproperty uchar green\
        \nproperty uchar blue\
        \nend_header\n";

    ofstream myfile;
    myfile.open(filename);
    myfile << header;
    for (int i = 0; i < points.size(); ++i)
    {
        Point3d cloudPoint = points[i];
        Point2d imagePoint = img_points[i];
        Vec3b color = image.at<Vec3b>(imagePoint);
        myfile << cloudPoint.x << " " << cloudPoint.y << " " << cloudPoint.z << " ";
        myfile << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << '\n';
    }
    myfile.close();
}

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

    // Mat drawing;
    // drawMatches(im1, feat1, im2, feat2, matches, drawing);
    // imshow("drawing", drawing);
    // waitKey();

    std::vector<Point2d> pts1, pts2;
    for (int i = 0; i < matches.size(); ++i)
    {
        pts1.push_back(feat1[matches[i].queryIdx].pt);
        pts2.push_back(feat2[matches[i].trainIdx].pt);
    }

    Mat F, E;
    vector<uchar> fundamental_inliers;
    MultiView::fundamental(pts1, pts2, F, fundamental_inliers);
    MultiView::essential(F, camera, camera, E);

    // drawEpipolarLines("epipolar lines", F, im1, im2, pts1, pts2);

    cout << "inliers: " << countNonZero(fundamental_inliers) << endl;

    vector<Point2d> best_pts1, best_pts2;
    Util::mask(pts1, fundamental_inliers, best_pts1);
    Util::mask(pts2, fundamental_inliers, best_pts2);

    pts1 = best_pts1;
    pts2 = best_pts2;

    vector<cv::Mat> rotations, translations;
    MultiView::get_rotation_and_translation(E, rotations, translations);

    int n = rotations.size();
    vector<vector<Point3d> > clouds(n);
    vector<vector<uchar> > inliers(n);
    vector<double> in_front_percent1(n), in_front_percent2(n);
    vector<double> projection_error1(n), projection_error2(n);

    for (int i = 0; i < n; ++i)
    {
        Mat rotation = rotations[i];
        Mat translation = translations[i];

        Mat no_rotation = Mat::eye(3, 3, CV_64F);
        Mat no_translation = Mat::zeros(3, 1, CV_64F);
        
        MultiView::triangulate(
            pts1, camera.matrix(),
            pts2, camera.matrix(),
            rotation, translation,
            clouds[i]);

        vector<Point2d> projected_pts1, projected_pts2;
        MultiView::project(clouds[i], no_rotation, no_translation, camera.matrix(), projected_pts1);
        MultiView::project(clouds[i], rotation, translation, camera.matrix(), projected_pts2);

        double in_front1 = 0.0, in_front2 = 0.0;
        double proj_err1 = 0.0, proj_err2 = 0.0;

        int num_points = pts1.size();
        for (int j = 0; j < num_points; ++j)
        {
            Point3d pt3d1, pt3d2;
            pt3d1 = clouds[i][j];
            MultiView::transform(pt3d1, rotation, translation, pt3d2);

            in_front1 += pt3d1.z > 0.0;
            in_front2 += pt3d2.z > 0.0;
            inliers[i].push_back(pt3d1.z > 0.0 and pt3d2.z > 0.0);

            proj_err1 += norm(Mat(pts1[j]), Mat(projected_pts1[j]));
            proj_err2 += norm(Mat(pts2[j]), Mat(projected_pts2[j]));
        }

        assert(num_points != 0);
        in_front1 /= num_points;
        in_front2 /= num_points;
        proj_err1 /= num_points;
        proj_err2 /= num_points;

        in_front_percent1[i] = in_front1;
        in_front_percent2[i] = in_front2;
        projection_error1[i] = proj_err1;
        projection_error2[i] = proj_err2;
    }

    int best_index = -1;
    for (int i = 0; i < 4; ++i)
    {
        if (in_front_percent1[i] > 0.75 &&
            in_front_percent2[i] > 0.75)
        {
            best_index = i;
            break;
        }
    }

    cout << best_index << endl;

    for (int i = 0; i < 4; ++i)
    {
        printf("%ld: (%f, %f) and (%f, %f)\n",
            clouds[i].size(),
            in_front_percent1[i],
            in_front_percent2[i],
            projection_error1[i],
            projection_error2[i]);
    }

    if (best_index == -1)
    {
        printf("Couldn't find a proper cloud.\n");
        return -1;
    }

    for (int i = 0; i < 4; ++i)
    {
        // vector<Point3d> cloud;
        // vector<Point2d> points;

        // Util::mask(clouds[i], inliers[i], cloud);
        // Util::mask(pts1, inliers[i], points);

        save_ply(im1, clouds[i], pts1, "cloud" + to_string(i)  + ".ply");
    }

    vector<Point3d> best_cloud;
    vector<Point2d> best_points;
    Util::mask(clouds[best_index], inliers[best_index], best_cloud);
    Util::mask(pts1, inliers[best_index], best_points);

    save_ply(im1, best_cloud, best_points, "cloud.ply");
}
