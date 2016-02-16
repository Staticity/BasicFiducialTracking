#include "VanillaTracker.hpp"

#include <highgui.hpp>
#include <calib3d.hpp>
#include <imgproc.hpp>

#include <assert.h>
#include <iostream>
#include <cstdio>

#include "Util.hpp"
#include <math.h>

#define DEBUG false

VanillaTracker::VanillaTracker() {}
VanillaTracker::~VanillaTracker() {}

bool VanillaTracker::triangulate(const Input& in, Output& out) const
{
    assert(in.pts1.size() == in.pts2.size());
    assert(in.camera.matrix().size() == cv::Size(3, 3));

    Args args;
    out = Output();

    if (!_computeFundamental (in, args, out)) return false;
    if (!_computeEssential   (in, args, out)) return false;
#if 0
    if (!_undistortPoints    (in, args, out)) return false;
#endif
    if (!_getCloud           (in, args, out)) return false;

    return true;
}

bool VanillaTracker::_computeFundamental(const Input& in, Args& args, Output& out) const
{
    static const int min_inliers = 100;

    assert(in.pts1.size() == in.pts2.size());
    int num_points = in.pts1.size();

    if (num_points == 0)
    {
        if (DEBUG) printf("[fundamental]: received no input points\n");
        return false;
    }

    double maxVal1, maxVal2;
    cv::minMaxIdx(in.pts1, 0, &maxVal1);
    cv::minMaxIdx(in.pts2, 0, &maxVal1);
    double maxVal = std::max(maxVal1, maxVal2); // not sure if passing same works.

    args.mask = std::vector<uchar>(num_points);
    out.fundamental = findFundamentalMat(in.pts1, in.pts2, cv::FM_RANSAC, 0.006 * maxVal, 0.99, args.mask);
    num_points = cv::countNonZero(args.mask);

    if (out.fundamental.empty())
    {
        if (DEBUG) printf("[fundamental]: fundamental matrix empty\n");
        return false;
    }

    if (num_points < min_inliers)
    {
        if (DEBUG) printf("[fundamental]: not enough inliers: %d/%d\n", num_points, min_inliers);
        return false;
    }

    return true;
}

bool VanillaTracker::_computeEssential(const Input& in, Args& args, Output& out) const
{
    static cv::Mat D = (cv::Mat_<double>(3, 3) << 1.0,  0.0,  0.0,
                                                  0.0,  1.0,  0.0,
                                                  0.0,  0.0,  0.0);

    static cv::Mat W = (cv::Mat_<double>(3, 3) << 0.0, -1.0,  0.0,
                                                  1.0,  0.0,  0.0,
                                                  0.0,  0.0,  1.0);

    static const double min_determinant = 1e-6;

    // Assume same camera
    cv::Mat camera_matrix = in.camera.matrix();
    out.essential = camera_matrix.t() * out.fundamental * camera_matrix;

    double e_determinant = determinant(out.essential);
    if (std::abs(e_determinant) > min_determinant)
    {
        if (DEBUG) printf("[essential]: expected determinant near 0, received: %f", e_determinant);
        return false;
    }

    // Compute essential matrix and its SVD decomp,
    // as well as the expected rotations and translations.
    // TODO: Make assertions return false with error message?
    {
        cv::SVD svd(out.essential);
        out.essential = svd.u * D * svd.vt; // Recompute E with D diagonal

        svd = cv::SVD(out.essential);

        // to be a "valid" essential matrix, we need the
        // rotation vector to be 1, we can accomplish
        // this by negating E and starting over.
        cv::Mat test_rotation = svd.u * W * svd.vt;
        if (Util::feq(cv::determinant(test_rotation), -1.0))
        {
            // NOTE:
            // This doesn't always work for some reason...
            // Sometimes we still get a rotation with -1
            out.essential = -out.essential;
            svd = cv::SVD(out.essential);
            // svd.vt.row(2) = -svd.vt.row(2);
        }

        cv::Mat r1 =  svd.u *   W   * svd.vt;
        cv::Mat r2 =  svd.u * W.t() * svd.vt;
        cv::Mat t1 =  svd.u.col(2);
        cv::Mat t2 = -svd.u.col(2);

        args.rotations.push_back(r1);
        args.rotations.push_back(r2);
        args.translations.push_back(t1);
        args.translations.push_back(t2);

        // Must have a proper rotation matrix with +-1 determinant
        assert(Util::feq(fabs(cv::determinant(args.rotations[0])) - 1.0, 0.0));

        // First two singular values must be equal
        assert(svd.w.type() == CV_64F);
        assert(Util::feq(svd.w.at<double>(0), svd.w.at<double>(1)));

        out.essentialU  = svd.u;
        out.essentialW  = svd.w;
        out.essentialVt = svd.vt;

        // Keep inliers.
        std::vector<int> indices(in.pts1.size());
        for (int i = 0; i < indices.size(); ++i)
            indices[i] = i;

        Util::retain(indices, args.mask, args.indices );
        Util::retain(in.pts1, args.mask, args.inliers1);
        Util::retain(in.pts2, args.mask, args.inliers2);
    }

    return true;
}

bool VanillaTracker::_undistortPoints(const Input& in, Args& args, Output& out) const
{
    cv::undistortPoints(args.inliers1, args.undistortedPts1, in.camera.matrix(), in.camera.distortion());
    cv::undistortPoints(args.inliers2, args.undistortedPts2, in.camera.matrix(), in.camera.distortion());

    return false; // unused
}

/**
 *  "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> VanillaTracker::_triangulatePoint(const cv::Point3d& u, const cv::Matx34d& P1,
                                                   const cv::Point3d& v, const cv::Matx34d& P2) const
{
    cv::Matx43d A(u.x * P1(2, 0) - P1(0, 0), u.x * P1(2, 1) - P1(0, 1), u.x * P1(2, 2) - P1(0, 2),      
                  u.y * P1(2, 0) - P1(1, 0), u.y * P1(2, 1) - P1(1, 1), u.y * P1(2, 2) - P1(1, 2),      
                  v.x * P2(2, 0) - P2(0, 0), v.x * P2(2, 1) - P2(0, 1), v.x * P2(2, 2) - P2(0, 2),   
                  v.y * P2(2, 0) - P2(1, 0), v.y * P2(2, 1) - P2(1, 1), v.y * P2(2, 2) - P2(1, 2));

    cv::Matx41d B(-(u.x * P1(2, 3) - P1(0, 3)),
                  -(u.y * P1(2, 3) - P1(1, 3)),
                  -(v.x * P2(2, 3) - P2(0, 3)),
                  -(v.y * P2(2, 3) - P2(1, 3)));

    cv::Mat_<double> X;
    cv::solve(A, B, X, cv::DECOMP_SVD);

    return X;
}

/**
 *  "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> VanillaTracker::_iterativeTriangulate(const cv::Point3d& u, const cv::Matx34d& P1,
                                                       const cv::Point3d& v, const cv::Matx34d& P2) const
{
    static const int max_iterations = 10;

    double wi1 = 1.0, wi2 = 1.0;
    cv::Mat_<double> X(4, 1);
    cv::Mat_<double> X_ = _triangulatePoint(u, P1, v, P2);

    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;

    for (int i = 0; i < max_iterations; ++i)
    {
        double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2) * X)(0);
        double p2x2 = cv::Mat_<double>(cv::Mat_<double>(P2).row(2) * X)(0);
        
        if(Util::feq(wi1, p2x1) && Util::feq(wi2, p2x2)) break;
        
        wi1 = p2x1;
        wi2 = p2x2;
        
        cv::Matx43d A((u.x*P1(2,0)-P1(0,0))/wi1, (u.x*P1(2,1)-P1(0,1))/wi1, (u.x*P1(2,2)-P1(0,2))/wi1,     
                      (u.y*P1(2,0)-P1(1,0))/wi1, (u.y*P1(2,1)-P1(1,1))/wi1, (u.y*P1(2,2)-P1(1,2))/wi1,     
                      (v.x*P2(2,0)-P2(0,0))/wi2, (v.x*P2(2,1)-P2(0,1))/wi2, (v.x*P2(2,2)-P2(0,2))/wi2, 
                      (v.y*P2(2,0)-P2(1,0))/wi2, (v.y*P2(2,1)-P2(1,1))/wi2, (v.y*P2(2,2)-P2(1,2))/wi2);
        
        cv::Mat_<double> B = (cv::Mat_<double>(4,1) << -(u.x * P1(2,3) - P1(0,3)) / wi1,
                                                       -(u.y * P1(2,3) - P1(1,3)) / wi1,
                                                       -(v.x * P2(2,3) - P2(0,3)) / wi2,
                                                       -(v.y * P2(2,3) - P2(1,3)) / wi2);
        
        solve(A, B, X_, cv::DECOMP_SVD);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X(3) = 1.0;
    }

    return X;
}

bool VanillaTracker::_getCloud(const Input& in, Args& args, Output& out) const
{
    static cv::Mat NoProjection     = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                                                                 0, 1, 0, 0,
                                                                 0, 0, 1, 0);
    static cv::Rect RotationROI     = cv::Rect(0, 0, 3, 3);
    static cv::Rect TranslationROI  = cv::Rect(3, 0, 1, 3);
    static const double min_percent = 0.75;
    static const double min_error   = 1.0;

    cv::Mat P1     = NoProjection.clone();
    cv::Mat K      = in.camera.matrix();
    cv::Mat K_inv  = K.inv();
    cv::Mat KP1    = K * P1;
    int num_points = args.inliers1.size();

    for (int i = 0; i < args.rotations.size(); ++i)
    {
        cv::Mat rotation = args.rotations[i];
        for (int j = 0; j < args.translations.size(); ++j)
        {
            cv::Mat translation = args.translations[j];

            // Verified correct copy
            cv::Mat P2 = cv::Mat::zeros(3, 4, P1.type());
            rotation.copyTo(P2(RotationROI));
            translation.copyTo(P2(TranslationROI));

            // cv::Mat KP2 = K * P2;

            int in_front = 0;
            double error_total = 0.0;
            for (int k = 0; k < num_points; ++k)
            {
                cv::Point2d pt1 = args.inliers1[k];
                cv::Point2d pt2 = args.inliers2[k];

                cv::Point3d u(pt1.x, pt1.y, 1.0);
                cv::Point3d v(pt2.x, pt2.y, 1.0);

                assert(Util::feq(pt1.x, u.x));
                assert(Util::feq(pt1.y, u.y));
                assert(Util::feq(pt2.x, v.x));
                assert(Util::feq(pt2.y, v.y));

                cv::Mat_<double> u_norm = K_inv * cv::Mat(u);
                cv::Mat_<double> v_norm = K_inv * cv::Mat(v);

                u.x = u_norm(0);
                u.y = u_norm(1);
                u.z = u_norm(2);
                
                v.x = v_norm(0);
                v.y = v_norm(1);
                v.z = v_norm(2);

                assert(Util::feq(u.x, u_norm(0)));
                assert(Util::feq(u.y, u_norm(1)));
                assert(Util::feq(u.z, u_norm(2)));

                assert(Util::feq(v.x, v_norm(0)));
                assert(Util::feq(v.y, v_norm(1)));
                assert(Util::feq(v.z, v_norm(2)));

                cv::Mat_<double> pt_3d = _iterativeTriangulate(u, P1, v, P2);

                std::cout << K << std::endl;
                std::cout << K_inv << std::endl;
                std::cout << pt_3d(0) << std::endl;
                std::cout << u << std::endl;
                std::cout << v << std::endl << std::endl;


                cv::Mat_<double> proj_pt1 = KP1 * pt_3d;

                if (Util::feq(proj_pt1(2), 0.0)) continue;
                cv::Point2d img_pt1(proj_pt1(0) / proj_pt1(2), proj_pt1(1) / proj_pt1(2));

                double error = norm(img_pt1 - pt1);//(norm(u_3d - pt_3d) + norm(v_3d - pt_3d)) / 2.0;
                error_total += error;

                bool is_in_front  = pt_3d(2) > 0.0;
                bool is_low_error = error < min_error;

                in_front += is_in_front;

                if (is_in_front && is_low_error)
                {
                    out.points.push_back(CloudPoint());

                    int out_index = out.points.size() - 1;
                    out.points[out_index].pt = cv::Point3d(pt_3d(0), pt_3d(1), pt_3d(2));
                    out.points[out_index].index = args.indices[k];
                }
            }

            const double front_percent = ((double)(in_front)) / std::max(1, num_points);
            const double error_avg     = error_total / std::max(1, num_points);

            // std::cout << front_percent * 100.0 << "%\t" << error_avg << std::endl;

            if (front_percent >= min_percent && error_avg < min_error)
            {
                out.rotation = rotation.clone();
                out.translation = translation.clone();
                out.visible_percent = front_percent;
                out.avg_reprojection_error = error_avg;
                return true;
            }
            else
            {
                out.points.clear();
            }
        }
    }

    if (DEBUG) printf("[get cloud]: No transformation satisfied the constraints.\n");

    return false;
}
