#include <opencv.hpp>
#include <xfeatures2d.hpp>

#include <cstdio>
#include <string>

#define DEBUG false

using namespace std;
using namespace cv;

struct CloudPoint
{
    cv::Point3d pt;
    cv::Point2d img_pt1, img_pt2;
};

void find_correspondences(const Mat& im1, const Mat& im2,
                            vector<Point>& pts1, vector<Point>& pts2)
{
    string matcher_type  = "BruteForce";

    // double t = getTickCount();

    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> feat1, feat2;
    detector->detect(im1, feat1);
    detector->detect(im2, feat2);

    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
    Mat descriptor1, descriptor2;
    extractor->compute(im1, feat1, descriptor1);
    extractor->compute(im2, feat2, descriptor2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_type);

    vector<vector<DMatch> > matches;
    matcher->knnMatch(descriptor1, descriptor2, matches, 2);

    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches[i].size() >= 2)
        {
            const float ratio = 0.8;
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                pts1.push_back(feat1[matches[i][0].queryIdx].pt);
                pts2.push_back(feat2[matches[i][0].trainIdx].pt);
            }
        }
    }

    // double time_spent = (((double)getTickCount()) - t) / getTickFrequency();
    // printf("[find_correspondences]: %0.4f seconds\n", time_spent);
    
    return time_spent;
}

bool fequals(double a, double b, double eps=1e-09)
{
    return fabs(a - b) < eps;
}

bool get_rotation_and_translation(const vector<Point>& pts1,
                                  const vector<Point>& pts2,
                                  const Mat& camera_matrix1,
                                  const Mat& camera_matrix2,
                                  const Mat& distortion_coeffs,
                                  Mat& rotation,
                                  Mat& translation,
                                  vector<CloudPoint>& cloud)
{
    assert(pts1.size() == pts2.size());
    assert(camera_matrix2.rows == 3);
    assert(camera_matrix1.rows == 3);
    assert(camera_matrix2.cols == 3);
    assert(camera_matrix1.cols == 3);

    int n_points = pts1.size();

    if (n_points < 8)
        return false;

    vector<uchar> is_inlier(n_points);
    Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3., 0.99, is_inlier);
    Mat E = camera_matrix2.t() * F * camera_matrix1;

    Mat D = (Mat_<double>(3, 3) << 1,  0,  0,
                                   0,  1,  0,
                                   0,  0,  0);

    Mat W = (Mat_<double>(3, 3) << 0, -1,  0,
                                   1,  0,  0,
                                   0,  0,  1);

    // Compute first SVD
    SVD svd(E, SVD::MODIFY_A);

    // Update E to include W as its svd diagonal
    E = svd.u * D * svd.vt;

    svd = SVD(E , SVD::MODIFY_A);

    // to be a "valid" essential matrix, we need the
    // rotation vector to be 1, we can accomplish
    // this by negating E and starting over.
    Mat test_rotation = svd.u * W * svd.vt;
    if (fequals(determinant(test_rotation), -1))
    {
        E = -E;
        svd = SVD(E , SVD::MODIFY_A);
    }

    vector<Mat> rotations, translations;
    rotations.push_back(svd.u *   W   * svd.vt);
    rotations.push_back(svd.u * W.t() * svd.vt);
    translations.push_back( svd.u.col(2));
    translations.push_back(-svd.u.col(2));

    // Must have a proper rotation matrix
    double rot_det = determinant(rotations[0]);
    assert(fequals(rot_det, -1.0) || fequals(rot_det, 1.0));

    // First two singular values must be equal
    assert(fequals(svd.w.at<double>(0), svd.w.at<double>(1)));

    // Strip out only the inliers
    vector<Point2d> left_inliers, right_inliers;
    vector<int> inlier_indices;

    double err = 0.0;
    for (int i = 0; i < n_points; ++i)
    {
        if (is_inlier[i])
        {
            left_inliers.push_back(pts1[i]);
            right_inliers.push_back(pts2[i]);
            inlier_indices.push_back(i);

            if (DEBUG)
            {
                Mat x2 = (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1);
                Mat x1 = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1);
                Mat dist = x2.t() * F * x1;
                err += dist.at<double>(0);
            }
        }
    }

    if (DEBUG)
    {
        err /= max((int)inlier_indices.size(), 1);
        cout << "average error: " << err << endl;
    }

    // Everything is pretty fuckin good up to this point

    vector<Point2d> left_npc_pts, right_npc_pts;
    undistortPoints(left_inliers ,  left_npc_pts, camera_matrix1, distortion_coeffs);
    undistortPoints(right_inliers, right_npc_pts, camera_matrix2, distortion_coeffs);

    if (DEBUG && 0)
    {
        for (int i = 0; i < left_npc_pts.size(); ++i)
        {
            cout << "Left: " << left_npc_pts[i] << " from " << left_inliers[i] << endl;
            cout << "Right: " << right_npc_pts[i] << right_inliers[i] << endl;
        }
    }

    // // Kind of bad, but let's reset n_points
    n_points = left_npc_pts.size();

    // No rotation and no translation
    Mat P1 = (Mat_<double>(3, 4) << 1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 1, 0);

    int index = 0;
    int best_index = -1;
    double best_percent = -1.0;
    for (int i = 0; i < rotations.size(); ++i)
    {
        for (int j = 0; j < translations.size(); ++j)
        {
            Mat P2 = Mat::zeros(3, 4, CV_64F);
            rotations[i].copyTo(P2(Rect(0, 0, 3, 3)));
            translations[j].copyTo(P2(Rect(3, 0, 1, 3)));

            assert(P1.type() == CV_64F);
            assert(P2.type() == CV_64F);

            Mat homogeneous_3d;
            triangulatePoints(P1,
                              P2,
                              left_npc_pts,
                              right_npc_pts,
                              homogeneous_3d);

            homogeneous_3d = homogeneous_3d.t();
            assert(homogeneous_3d.type() == CV_64F);
            assert(homogeneous_3d.rows == n_points);
            assert(homogeneous_3d.cols == 4);
            assert(homogeneous_3d.clone().checkVector(4) >= 0);

            // Doesn't zero out when w = 0...
            // vector<Point3d> points3d;
            // convertPointsFromHomogeneous(homogeneous_3d, points3d);
            // assert(points3d.size() == n_points);

            vector<Point3d> points3d;
            vector<int> point_indices;
            int in_front = 0;
            const Vec4d* pts4 = homogeneous_3d.ptr<Vec4d>();
            for (int k = 0; k < n_points; ++k)
            {
                double x = pts4[k][0];
                double y = pts4[k][1];
                double z = pts4[k][2];
                double w = pts4[k][3];
                double scale = 0.0;

                if (!fequals(w, 0))
                {
                    scale = 1.0 / w;
                    x *= scale;
                    y *= scale;
                    z *= scale;

                    points3d.push_back(Point3d(x, y, z));
                    point_indices.push_back(inlier_indices[k]);
                    in_front += z > 0.0;
                }
            }

            const double min_in_front_percent = 0.75;
            const double percent_in_front = ((double) in_front) / max(n_points, 1);

            if (percent_in_front >= min_in_front_percent)
            {
                Mat R = P1(Rect(0, 0, 3, 3));
                Vec3d rvec(0, 0, 0); // Rodrigues identity rotation
                Vec3d tvec(0, 0, 0); // no translation

                vector<Point2d> reprojected_pts;
                projectPoints(points3d,
                              rvec,
                              tvec,
                              camera_matrix1,
                              distortion_coeffs,
                              reprojected_pts);

                vector<Point2d> original_pts(reprojected_pts.size());
                for (int k = 0; k < reprojected_pts.size(); ++k)
                {
                    original_pts[k] = pts1[point_indices[k]];
                }

                double avg_error = cv::norm(Mat(reprojected_pts), Mat(original_pts), NORM_L2) / max((double) reprojected_pts.size(), 1.0);

                if (DEBUG)
                    cout << avg_error << endl;

                if (avg_error <= 10.0)
                {
                    if (percent_in_front > best_percent)
                    {                        
                        best_percent = percent_in_front;
                        best_index = index;

                        cloud = vector<CloudPoint>(reprojected_pts.size());
                        for (int k = 0; k < reprojected_pts.size(); ++k)
                        {
                            int pt_index = point_indices[k];
                            cloud[k].pt = points3d[k];
                            cloud[k].img_pt1 = pts1[pt_index];
                            cloud[k].img_pt2 = pts2[pt_index];

                            if (DEBUG)
                            {
                                cout << k << " -> " << point_indices[k] << endl;
                                cout << cloud[k].pt << endl;
                                cout << cloud[k].img_pt1 << endl;
                                cout << cloud[k].img_pt2 << endl << endl;
                            }
                        }
                    }
                }
            }

            ++index;
            if (DEBUG)
                cout << in_front << " / " << n_points << endl;
        }
    }


    if (best_index == -1)
        return false;

    return true;
}

// assumes no skew
void update_camera_matrix_resize(Mat& camera_matrix, Size old_size, Size new_size)
{
    assert(old_size.width * new_size.height == old_size.height * new_size.width);
    assert(old_size.width != 0 && old_size.height != 0);

    double width_ratio = ((double) new_size.width) / old_size.width;
    double height_ratio = ((double) new_size.height) / old_size.height;

    camera_matrix.at<double>(0, 0) *= width_ratio;
    camera_matrix.at<double>(1, 1) *= height_ratio;
    camera_matrix.at<double>(0, 2) = new_size.width * 0.5;
    camera_matrix.at<double>(1, 2) = new_size.height * 0.5;
}

int main(int argc, char** argv)
{
    int index = 0;
    if (argc > 1)
        index = atoi(argv[1]);

    string camera_calib_filename = "calibration_data/out_camera_data.xml";
    FileStorage fs(camera_calib_filename, FileStorage::READ);

    Mat camera_matrix, distortion_coeffs;
    int camera_width, camera_height;
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coeffs;
    fs["image_width"] >> camera_width;
    fs["image_height"] >> camera_height;

    int width = 1280;
    int height = 720;

    update_camera_matrix_resize(
        camera_matrix,
        Size(camera_width, camera_height),
        Size(width, height));

    VideoCapture vc(index);

    if (!vc.isOpened()) return 0;

    vc.set(CV_CAP_PROP_FRAME_WIDTH, width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    Mat im1, im2;
    vc >> im1;
    while (vc.isOpened())
    {
        vc >> im1;
        imshow("im1", im1);
        if (waitKey(20) == 27) break;
    }
    while (vc.isOpened())
    {
        vc >> im2;
        imshow("im2", im2);
        if (waitKey(20) == 27) break;
    }
    // Mat im1 = imread("test1.jpg");
    // Mat im2 = imread("test2.jpg");

    vector<Point> pts1, pts2;
    find_correspondences(im1, im2, pts1, pts2);
    assert(pts1.size() == pts2.size());

    
    Mat rotation, translation;
    vector<CloudPoint> cloud;
    bool found = get_rotation_and_translation(
        pts1,
        pts2,
        camera_matrix,
        camera_matrix,
        distortion_coeffs,
        rotation,
        translation,
        cloud);

    // printf("Found cloud: %d\n", found);
    // cout << cloud.size() << endl;

    imshow("left", im1);
    RNG rng(0);
    if (found)
    {
        for (int i = 0; i < cloud.size(); ++i)
        {
            Vec3b color = im1.at<Vec3b>(cloud[i].img_pt1.y, cloud[i].img_pt1.x);
            printf("%f %f %f %d %d %d\n", cloud[i].pt.x, cloud[i].pt.y, cloud[i].pt.z,
                                          color[0], color[1], color[2]);

            // cout << cloud[i].img_pt1 << endl;
            circle(im1, cloud[i].img_pt1, 5, Scalar(0, 255, 0), 3);
        }
    }
    imshow("left", im1);
    waitKey();

    if (0)
    {
        int n = pts1.size();
        // barf way to keep infliers lol
        vector<uchar> is_inlier(n);
        Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3., 0.99, is_inlier);

        // for (int i = 0; i < n; ++i)
        // {
        //     Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        //     circle(im1, pts1[i], 10, color, 3);
        //     circle(im2, pts2[i], 10, color, 3);
        // }

        Mat joined_image  = Mat::zeros(max(im1.rows, im2.rows), im1.cols + im2.cols, im1.type());
        Rect left_roi  = Rect(0, 0, im1.cols, im1.rows);
        Rect right_roi = Rect(im1.cols, 0, im2.cols, im2.rows);

        Mat left_portion  = Mat(joined_image, left_roi);
        Mat right_portion = Mat(joined_image, right_roi);

        im1.copyTo(left_portion);
        im2.copyTo(right_portion);

        for (int i = 0; i < n; ++i)
        {
            if (!is_inlier[i]) continue;
            if (rng.uniform(1, 100) > 5) continue;
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            Point p1 = pts1[i];
            Point p2 = pts2[i] + Point(im1.cols, 0);
            line(joined_image, p1, p2, color, 2);
        }

        imshow("im", joined_image);
        waitKey();
    }
}
