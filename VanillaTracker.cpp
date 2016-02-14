#include "VanillaTracker.hpp"

#include <calib3d.hpp>
#include <imgproc.hpp>

#include <assert.h>
#include <iostream>

#include "Util.hpp"

VanillaTracker::VanillaTracker() {}
VanillaTracker::~VanillaTracker() {}

bool VanillaTracker::triangulate(const Input& in, Output& out) const
{
    assert(in.pts1.size() == in.pts2.size());
    assert(in.camera.matrix().size() == cv::Size(3, 3));

    Args args;
    out = Output();

    std::cout << "fundamental" << std::endl;
    if (!_computeFundamental (in, args, out)) return false;
    std::cout << "essential" << std::endl;
    if (!_computeEssential   (in, args, out)) return false;
#if 0
    if (!_undistortPoints    (in, args, out)) return false;
#endif
    std::cout << "get cloud" << std::endl;
    if (!_getCloud           (in, args, out)) return false;

    return true;
}

bool VanillaTracker::_computeFundamental(const Input& in, Args& args, Output& out) const
{
    static int min_inliers = 100;
    int num_points = in.pts1.size();

    // double maxVal;
    // cv::minMaxIdx(in.pts1, 0, &maxVal);
    args.mask = std::vector<uchar>(num_points);
    out.fundamental = findFundamentalMat(in.pts1, in.pts2, cv::FM_RANSAC, 0.1/*0.006 * maxVal*/, 0.99, args.mask);

    num_points = cv::countNonZero(args.mask);

    if (num_points < min_inliers) return false;
    if (out.fundamental.empty()) return false;

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

    // Assume same camera
    cv::Mat camera_matrix = in.camera.matrix();
    out.essential = camera_matrix.t() * out.fundamental * camera_matrix;

    // Keep this......?
    if (!Util::feq(determinant(out.essential), 0)) return false;// std::cout << determinant(out.essential) << std::endl;
    assert(out.essential.type() == CV_64F);

    // Compute essential matrix and its SVD decomp,
    // as well as the expected rotations and translations.
    // TODO: Make assertions return false with error message?
    {
        cv::SVD svd(out.essential, cv::SVD::MODIFY_A);
        out.essential = svd.u * D * svd.vt; // Recompute E with D diagonal

        svd = cv::SVD(out.essential , cv::SVD::MODIFY_A);

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
            svd = cv::SVD(out.essential , cv::SVD::MODIFY_A);
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

    return true;
}

bool VanillaTracker::_triangulate(const Input& in, Args& args, Output& out) const
{
    return false;
}

/**
 *  "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> VanillaTracker::_triangulatePoint(const cv::Point3d& u, const cv::Matx34d& P1,
                                                   const cv::Point3d& v, const cv::Matx34d& P2) const
{
    cv::Matx43d A(u.x * P1(2,0) - P1(0,0), u.x * P1(2,1) - P1(0,1), u.x * P1(2,2) - P1(0,2),      
                  u.y * P1(2,0) - P1(1,0), u.y * P1(2,1) - P1(1,1), u.y * P1(2,2) - P1(1,2),      
                  v.x * P2(2,0) - P2(0,0), v.x * P2(2,1) - P2(0,1), v.x * P2(2,2) - P2(0,2),   
                  v.y * P2(2,0) - P2(1,0), v.y * P2(2,1) - P2(1,1), v.y * P2(2,2) - P2(1,2));

    cv::Matx41d B(-(u.x*P1(2,3) - P1(0,3)),
                  -(u.y*P1(2,3) - P1(1,3)),
                  -(v.x*P2(2,3) - P2(0,3)),
                  -(v.y*P2(2,3) - P2(1,3)));

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
        
        //reweight equations and solve
        cv::Matx43d A((u.x*P1(2,0)-P1(0,0))/wi1, (u.x*P1(2,1)-P1(0,1))/wi1, (u.x*P1(2,2)-P1(0,2))/wi1,     
                      (u.y*P1(2,0)-P1(1,0))/wi1, (u.y*P1(2,1)-P1(1,1))/wi1, (u.y*P1(2,2)-P1(1,2))/wi1,     
                      (v.x*P2(2,0)-P2(0,0))/wi2, (v.x*P2(2,1)-P2(0,1))/wi2, (v.x*P2(2,2)-P2(0,2))/wi2, 
                      (v.y*P2(2,0)-P2(1,0))/wi2, (v.y*P2(2,1)-P2(1,1))/wi2, (v.y*P2(2,2)-P2(1,2))/wi2);
        
        cv::Mat_<double> B = (cv::Mat_<double>(4,1) <<    -(u.x * P1(2,3) - P1(0,3)) / wi1,
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
    static const double min_error   = 10.0;

    cv::Mat P1 = NoProjection.clone();
    int num_points = args.inliers1.size();

    for (int i = 0; i < args.rotations.size(); ++i)
    {
        cv::Mat rotation    = args.rotations[i];
        for (int j = 0; j < args.translations.size(); ++j)
        {
            cv::Mat translation = args.translations[j];

            cv::Mat P2 = cv::Mat::zeros(3, 4, P1.type());
            rotation.copyTo(P2(RotationROI));
            translation.copyTo(P2(TranslationROI));
#if 0

            cv::Mat pts_homog_3d(1, num_points, CV_64FC4);
            std::vector<cv::Point3d> pts_3d;
            cv::triangulatePoints(P1, P2, args.undistortedPts1.reshape(1, 2), args.undistortedPts2.reshape(1, 2), pts_homog_3d);
            cv::convertPointsFromHomogeneous(pts_homog_3d.reshape(4, 1), pts_3d);


            // NOTE:
            // This assumes that no points have w = 0, since
            // as of 2/13/2016 convertPointsFromHomogenous
            // doesn't zero out the vector if w = 0.
            std::vector<unsigned char> status(num_points);
            for (int k = 0; k < num_points; ++k)
            {
                status[k] = (pts_3d[k].z > 0.0);
            }

            int num_in_front  = cv::countNonZero(status);
            double front_percent = ((double)(num_in_front)) / std::max(1, num_points);

            std::cout << front_percent << std::endl;

            // Not enough points in front
            if (front_percent < min_percent) continue;

            cv::Vec3d rvec, tvec;
            Rodrigues(P1(RotationROI), rvec);
            tvec = P1(TranslationROI);

            std::vector<cv::Point2d> reprojected_pts1;
            cv::projectPoints(pts_3d, rvec, tvec, in.camera.matrix(), in.camera.distortion(), reprojected_pts1);

            const double error_total = cv::norm(cv::Mat(reprojected_pts1), cv::Mat(args.inliers1), cv::NORM_L2); 
            const double error_avg   = error_total / std::max((int)reprojected_pts1.size(), 1);

            if (error_avg > min_error) continue;

            for (int k = 0; k < num_points; ++k)
            {
                const double error = cv::norm(args.inliers1[k] - reprojected_pts1[k]);
                status[k] &= (error < 2 * min_error);
            }

            out.rotation               = rotation;
            out.translation            = translation;
            out.visible_percent        = front_percent;
            out.avg_reprojection_error = error_avg;

            out.points.clear();
            for (int k = 0; k < num_points; ++k)
            {
                if (status[k])
                {
                    out.points.push_back(CloudPoint());
                    int out_index = out.points.size() - 1;
                    out.points[out_index].pt = pts_3d[k];
                    out.points[out_index].index = args.indices[k];
                }
            }
#else
            int in_front = 0;
            double error_total = 0.0;
            cv::Mat camera     = in.camera.matrix();
            cv::Mat camera_inv = camera.inv();
            for (int k = 0; k < num_points; ++k)
            {
                cv::Point2d pt1 = args.inliers1[k];
                cv::Point2d pt2 = args.inliers2[k];

                cv::Point3d u(pt1.x, pt1.y, 1.0);
                cv::Point3d v(pt2.x, pt2.y, 1.0);
                
                cv::Mat_<double> u_3d    = camera_inv * cv::Mat(u);
                cv::Mat_<double> v_3d    = camera_inv * cv::Mat(v);
                cv::Mat_<double> pt_3d   = _iterativeTriangulate(u, P1, v, P2);

                cv::Mat_<double> proj_pt = camera * P1 * pt_3d;

                assert(!Util::feq(pt_3d(2), 0.0));
                cv::Point2d img_pt1(proj_pt(0) / proj_pt(2), proj_pt(1) / proj_pt(2));

                double error = norm(pt1 - img_pt1);
                error_total += error;

                bool is_in_front  = pt_3d(2) > 0.0;
                bool is_low_error = error < 2 * min_error;

                in_front += is_in_front;

                if (is_in_front && is_low_error)
                {
                    out.points.push_back(CloudPoint());
                    out.points[i].pt = cv::Point3d(pt_3d(0), pt_3d(1), pt_3d(2));
                    out.points[i].index = args.indices[k];
                }
            }

            const double front_percent = ((double)(in_front)) / std::max(1, num_points);
            const double error_avg     = error_total / std::max(1, num_points);

            std::cout << (front_percent * 100.0) << "%" << " " << error_avg << std::endl;

            if (front_percent < min_percent && error_avg < min_error)
            {
                out.rotation = rotation;
                out.translation = translation;
                out.visible_percent = front_percent;
                out.avg_reprojection_error = error_avg;
                return true;
            }
            else
            {
                out.points.clear();
            }
#endif
        }
    }
    std::cout << std::endl;

    return false;
}
