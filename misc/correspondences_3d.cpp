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

bool find_correspondences(const Mat& im1, const Mat& im2,
                          vector<KeyPoint>& feat1, vector<KeyPoint>& feat2,
                          vector<DMatch>& matches)
{
    // printf("We made it boys\n");
    string matcher_type  = "BruteForce";

    double t = getTickCount();

    /*
    Ptr<AKAZE> akaze = AKAZE::create();
    Mat descriptor1, descriptor2;
    akaze->detectAndCompute(im1, noArray(), feat1, descriptor1);
    akaze->detectAndCompute(im2, noArray(), feat2, descriptor2);*/

    
    // printf("About to detect...\n");
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(20);//xfeatures2d::SIFT::create();
    detector->detect(im1, feat1);
    detector->detect(im2, feat2);

    if (feat1.empty() || feat2.empty())
        return false;

    // cout << feat1.size() << " >>><<< " << feat2.size() << endl;

    if (DEBUG)
        printf("Features: %0.4f seconds\n", (getTickCount() - t) / getTickFrequency());
    t = getTickCount();

    Ptr<DescriptorExtractor> extractor = /*ORB::create();*/xfeatures2d::SIFT::create();
    Mat descriptor1, descriptor2;
    extractor->compute(im1, feat1, descriptor1);
    extractor->compute(im2, feat2, descriptor2);

    if (feat1.empty() || feat2.empty())
        return false;

    if (DEBUG)
        printf("Descriptors: %0.4f seconds\n", (getTickCount() - t) / getTickFrequency());
    t = getTickCount();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_type);
    vector<vector<DMatch> > match_array;

    matcher->knnMatch(descriptor1, descriptor2, match_array, 2);

    if (DEBUG)
        printf("Performing ratio test and shit\n");

    for (int i = 0; i < match_array.size(); ++i)
    {
        if (match_array[i].size() >= 2)
        {
            const float ratio = 0.8;
            if (match_array[i][0].distance < ratio * match_array[i][1].distance)
            {
                matches.push_back(match_array[i][0]);
            }
        }
    }

    if (DEBUG)
        printf("Matches: %0.4f seconds\n", (getTickCount() - t) / getTickFrequency());

    if (DEBUG)
    {
        // printf("Begin drawing matches:");
        // Mat drawing;
        // drawMatches(im1, feat1, im2, feat2, matches, drawing);
        // imshow("matches", drawing);
        // waitKey();
        // destroyWindow("matches");
        printf(" Done\n");
    }
    // double time_spent = (((double)getTickCount()) - t) / getTickFrequency();
    // printf("[find_correspondences]: %0.4f seconds\n", time_spent);
    // return time_spent;

    return true;
}

bool fequals(double a, double b, double eps=1e-09)
{
    return fabs(a - b) < eps;
}

bool get_rotation_and_translation(const vector<Point2d>& pts1,
                                  const vector<Point2d>& pts2,
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
    assert(camera_matrix1.type() == CV_64F);
    assert(camera_matrix2.type() == CV_64F);
    assert(distortion_coeffs.type() == CV_64F);

    int n_points = pts1.size();

    if (n_points < 8)
        return false;

    if (DEBUG)
        printf("Okay, we're going somewhere\n");

    vector<uchar> is_inlier(n_points);
    Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3.0, 0.99, is_inlier);

    if (F.rows == 0 || F.cols == 0)
        return false;

    assert(F.type() == CV_64F);

    if (DEBUG)
        printf("Created fundamental matrix\n");
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

    if (DEBUG)
        printf("Created essential matrix\n");

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
                err += abs(dist.at<double>(0));
            }
        }
    }

    if (DEBUG)
    {
        err /= max((int)inlier_indices.size(), 1);
        cout << "average fundamental error: " << err << endl;
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

                vector<Point2d> reprojected_pts1;
                projectPoints(points3d,
                              rvec,
                              tvec,
                              camera_matrix1,
                              distortion_coeffs,
                              reprojected_pts1);

                R = P2(Rect(0, 0, 3, 3));
                Rodrigues(R, rvec);
                tvec = P2(Rect(3, 0, 1, 3));

                vector<Point2d> reprojected_pts2;
                projectPoints(points3d,
                              rvec,
                              tvec,
                              camera_matrix1,
                              distortion_coeffs,
                              reprojected_pts2);

                assert(reprojected_pts1.size() == reprojected_pts2.size());
                vector<Point2d> original_pts1(reprojected_pts1.size());
                vector<Point2d> original_pts2(reprojected_pts2.size());
                for (int k = 0; k < reprojected_pts1.size(); ++k)
                {
                    original_pts1[k] = pts1[point_indices[k]];
                    original_pts2[k] = pts2[point_indices[k]];
                }

                double avg_error1 = cv::norm(Mat(reprojected_pts1), Mat(original_pts1), NORM_L2) / max((double) reprojected_pts1.size(), 1.0);
                double avg_error2 = cv::norm(Mat(reprojected_pts2), Mat(original_pts2), NORM_L2) / max((double) reprojected_pts2.size(), 1.0);

                if (DEBUG)
                {
                    cout << avg_error1 << endl;
                    cout << avg_error2 << endl;
                }

                if (avg_error1 <= 10.0 && avg_error2 <= 10.0)
                {
                    if (percent_in_front > best_percent)
                    {                        
                        best_percent = percent_in_front;
                        best_index   = index;
                        rotation     = rotations[i];
                        translation  = translations[i];
                        cloud = vector<CloudPoint>(reprojected_pts1.size());
                        for (int k = 0; k < reprojected_pts1.size(); ++k)
                        {
                            int pt_index = point_indices[k];
                            cloud[k].pt = points3d[k];
                            cloud[k].img_pt1 = pts1[pt_index];
                            cloud[k].img_pt2 = pts2[pt_index];

                            if (DEBUG & 0)
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

    int width = 640;
    int height = 360;

    update_camera_matrix_resize(
        camera_matrix,
        Size(camera_width, camera_height),
        Size(width, height));

    VideoCapture vc(index);

    if (!vc.isOpened()) return 0;

    vc.set(CV_CAP_PROP_FRAME_WIDTH, width);
    vc.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    Mat im1, im2;
    im1 = imread("images/Lenna.png");
    while (vc.isOpened())
    {
        vc >> im2;
        vector<Point2d> pts1, pts2;
        std::vector<KeyPoint> feat1, feat2;
        std::vector<DMatch> matches;
        double t = getTickCount();
        bool found = find_correspondences(im1, im2, feat1, feat2, matches);
        printf("[find_correspondences]: %0.4f seconds\n", (getTickCount() - t) / getTickFrequency());

        // cout << matches.size() << " " << feat1.size() << " " << feat2.size() << endl;
        for (int i = 0; i < matches.size(); ++i)
        {
            // cout << i << endl;
            // cout << matches[i].queryIdx << " " << matches[i].trainIdx << endl;
            pts1.push_back(Point2d(feat1[matches[i].queryIdx].pt));
            pts2.push_back(Point2d(feat2[matches[i].trainIdx].pt));
        }

        assert(pts1.size() == pts2.size());
        
        Mat rotation, translation;
        vector<CloudPoint> cloud;

        // cout << "get_rotation_and_translation" << endl;
        // double t = getTickCount();
        if (found)
        {
            t = getTickCount();
            found = get_rotation_and_translation(
                pts1,
                pts2,
                camera_matrix,
                camera_matrix,
                distortion_coeffs,
                rotation,
                translation,
                cloud);
            printf("[get_rotation_and_translation]: %0.4f seconds\n", (getTickCount() - t) / getTickFrequency());
        }

        if (found)
        {

            // cout << "found some good shit" << endl;
            vector<char> mask;
            Mat homography = findHomography(pts1, pts2, CV_RANSAC, 3, mask);
            Mat points = (Mat_<double>(4, 3) <<        0,        0, 1,
                                                im1.cols,        0, 1,
                                                im1.cols, im1.rows, 1,
                                                       0, im1.rows, 1);

            // Mat drawing;
            // drawMatches(im1, feat1, im2, feat2, matches, drawing,
            //             Scalar::all(-1), Scalar::all(-1),
            //             mask, DrawMatchesFlags::DEFAULT);

            // imshow("drawing", drawing);
            // waitKey();

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

            // for (int i = 0; i < 4; ++i)
            // {
            //     int j = (i + 1) % 4;
            //     line(im2, quad[i], quad[j], Scalar(0, 255, 0), 3);
            // }

            double cube_points[8][3] =
            {
                {-1.0,  1.0,  0.0f},
                { 1.0,  1.0,  0.0f},
                { 1.0, -1.0,  0.0f},
                {-1.0, -1.0,  0.0f},
                {-1.0,  1.0,  1.0f},
                { 1.0,  1.0,  1.0f},
                { 1.0, -1.0,  1.0f},
                {-1.0, -1.0,  1.0f},
            };

            Mat cube_object_coordinates = Mat(8, 3, CV_64F, cube_points);

            cout << rotation << "\n" << translation << "\n\n" << endl;
            Mat projected_points;
            projectPoints(
                cube_object_coordinates,
                rotation,
                translation,
                camera_matrix,
                distortion_coeffs,
                projected_points);

            vector<Point> cube_projected_points(8);
            for (int i = 0; i < 8; ++i)
            {
                cube_projected_points[i] = Point2f(0.0, 0.0);
                cube_projected_points[i].x = projected_points.at<double>(i, 0);
                cube_projected_points[i].y = projected_points.at<double>(i, 1);
            }
            
            Scalar color = Scalar(0, 255, 0);
            for (int i = 0; i < 4; ++i)
            {
                int i1 = i;
                int i2 = (i + 1) % 4;
                int i3 = i1 + 4;
                int i4 = i2 + 4;
                line(im2, cube_projected_points[i1], cube_projected_points[i2], color, 3);
                line(im2, cube_projected_points[i3], cube_projected_points[i4], color, 3);
                line(im2, cube_projected_points[i1], cube_projected_points[i3], color, 3);
            }
        }
        // else
        imshow("im2", im2);

        if (waitKey(20) == 'r') vc >> im1;
        if (waitKey(20) == 27) break;
    }
}
