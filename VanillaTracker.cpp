#include "VanillaTracker.hpp"

#include "Util.hpp"

#include <calib3d.hpp>

#include <assert.h>

VanillaTracker::VanillaTracker() {}
VanillaTracker::~VanillaTracker() {}

bool VanillaTracker::triangulate(const Input& in, Output& out) const
{
    assert(in.feat1.size() == in.feat1.size());
    assert(in.camera.matrix().size() == cv::Size(3, 3));

    Args args;

    if (!_computeFundamental(in, args, out)) return false;
    if (!_computeEssential(in, args, out)) return false;
    if (!_undistortPoints(in, args, out)) return false;
    if (!_triangulate(in, args, out)) return false;
    if (!_testError(in, args, out)) return false;

    return true;
}

bool VanillaTracker::_computeFundamental(const Input& in, Args& args, Output& out) const
{
    int num_points = in.feat1.size();

    if (num_points < 8) return false;

    std::vector<cv::Point> pts1, pts2;
    Util::toPoints(in.feat1, pts1);
    Util::toPoints(in.feat2, pts2);

    out.mask = std::vector<unsigned char>(num_points);
    out.fundamental = findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3., 0.99, out.mask);

    if (out.fundamental.empty()) return false;

    return false;
}

bool VanillaTracker::_computeEssential(const Input& in, Args& args, Output& out) const
{
    static Mat D = (Mat_<double>(3, 3) << 1,  0,  0,
                                          0,  1,  0,
                                          0,  0,  0);

    static Mat W = (Mat_<double>(3, 3) << 0, -1,  0,
                                          1,  0,  0,
                                          0,  0,  1);

    // Assume same camera
    Mat camera_matrix = in.camera.matrix();
    out.essential = camera_matrix.t() * out.fundamental * camera_matrix;
    
    // Compute essential matrix and its SVD decomp,
    // as well as the expected rotations and translations.
    // TODO: Make assertions return false with error message?
    {
        SVD svd(E, SVD::MODIFY_A);
        out.essential = svd.u * D * svd.vt; // Recompute E with D diagonal

        svd = SVD(out.essential , SVD::MODIFY_A);

        // to be a "valid" essential matrix, we need the
        // rotation vector to be 1, we can accomplish
        // this by negating E and starting over.
        Mat test_rotation = svd.u * W * svd.vt;
        if (Util::feq(determinant(test_rotation), -1.0))
        {
            out.essential = -out.essential;
            svd = SVD(out.essential , SVD::MODIFY_A);
        }

        Mat r1 =  svd.u *   W   * svd.vt;
        Mat r2 =  svd.u * W.t() * svd.vt;
        Mat t1 =  svd.u.col(2);
        Mat t2 = -svd.u.col(2);

        args.rotations.push_back(r1);
        args.rotations.push_back(r2);
        args.translations.push_back(t1);
        args.translations.push_back(t2);

        // Must have a proper rotation matrix with 1 determinant
        assert(fequals(determinant(rotations[0]), 1.0));

        // First two singular values must be equal
        assert(fequals(svd.w.at<double>(0), svd.w.at<double>(1)));
    }

    // Keep inliers.
    {
        Util::mask(in.feat1, args.mask, args.inliers1);
        Util::mask(in.feat2, args.mask, args.inliers2);
    }

    return true;
}

bool VanillaTracker::_undistortPoints(const Input& in, Args& args, Output& out) const
{
    return false;
}

bool VanillaTracker::_triangulate(const Input& in, Args& args, Output& out) const
{
    return false;
}

bool VanillaTracker::_testError(const Input& in, Args& args, Output& out) const
{
    return false;
}

/*
bool get_rotation_and_translation(const vector<Point2d>& pts1,
                                  const vector<Point2d>& pts2,
                                  const Mat& camera_matrix1,
                                  const Mat& camera_matrix2,
                                  const Mat& distortion_coeffs,
                                  Mat& rotation,
                                  Mat& translation,
                                  vector<CloudPoint>& cloud)
{

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
*/