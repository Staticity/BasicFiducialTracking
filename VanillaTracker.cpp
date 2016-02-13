#include "VanillaTracker.hpp"

#include <calib3d.hpp>
#include <imgproc.hpp>

#include <assert.h>

#include "Util.hpp"

VanillaTracker::VanillaTracker() {}
VanillaTracker::~VanillaTracker() {}

bool VanillaTracker::triangulate(const Input& in, Output& out) const
{
    assert(in.feat1.size() == in.feat1.size());
    assert(in.camera.matrix().size() == cv::Size(3, 3));

    Args args;
    out = Output();

    if (!_computeFundamental (in, args, out)) return false;
    if (!_computeEssential   (in, args, out)) return false;
    if (!_undistortPoints    (in, args, out)) return false;
    if (!_triangulate        (in, args, out)) return false;

    return true;
}

bool VanillaTracker::_computeFundamental(const Input& in, Args& args, Output& out) const
{
    int num_points = in.feat1.size();

    if (num_points < 8) return false;

    std::vector<cv::Point2d> pts1, pts2;
    Util::toPoints(in.feat1, pts1);
    Util::toPoints(in.feat2, pts2);

    args.mask = std::vector<unsigned char>(num_points);
    out.fundamental = findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3., 0.99, args.mask);

    if (out.fundamental.empty()) return false;

    return false;
}

bool VanillaTracker::_computeEssential(const Input& in, Args& args, Output& out) const
{
    static cv::Mat D = (cv::Mat_<double>(3, 3) << 1,  0,  0,
                                                  0,  1,  0,
                                                  0,  0,  0);

    static cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1,  0,
                                                  1,  0,  0,
                                                  0,  0,  1);

    // Assume same camera
    cv::Mat camera_matrix = in.camera.matrix();
    out.essential = camera_matrix.t() * out.fundamental * camera_matrix;

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
            out.essential = -out.essential;
            svd = cv::SVD(out.essential , cv::SVD::MODIFY_A);
        }

        cv::Mat r1 =  svd.u *   W   * svd.vt;
        cv::Mat r2 =  svd.u * W.t() * svd.vt;
        cv::Mat t1 =  svd.u.col(2);
        cv::Mat t2 = -svd.u.col(2);

        args.rotations.push_back(r1);
        args.rotations.push_back(r2);
        args.translations.push_back(t1);
        args.translations.push_back(t2);

        // Must have a proper rotation matrix with 1 determinant
        assert(Util::feq(cv::determinant(args.rotations[0]), 1.0));

        // First two singular values must be equal
        assert(svd.w.type() == CV_64F);
        assert(Util::feq(svd.w.at<double>(0), svd.w.at<double>(1)));

        out.essentialU = svd.u;
        out.essentialW = svd.w;
        out.essentialVt = svd.vt;

        // Keep inliers.
        std::vector<int> indices(in.feat1.size());
        for (int i = 0; i < indices.size(); ++i)
            indices[i] = i;

        Util::retain(indices , args.mask, args.indices );
        Util::retain(in.feat1, args.mask, args.inliers1);
        Util::retain(in.feat2, args.mask, args.inliers2);
    }

    return true;
}

bool VanillaTracker::_undistortPoints(const Input& in, Args& args, Output& out) const
{
    std::vector<cv::Point2d> pts1, pts2;
    Util::toPoints(args.inliers1, pts1);
    Util::toPoints(args.inliers2, pts2);

    cv::undistortPoints(pts1, args.undistortedPts1, in.camera.matrix(), in.camera.distortion());
    cv::undistortPoints(pts2, args.undistortedPts2, in.camera.matrix(), in.camera.distortion());

    return true;
}

bool VanillaTracker::_triangulate(const Input& in, Args& args, Output& out) const
{
    static cv::Mat NoProjection     = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                                                                 0, 1, 0, 0,
                                                                 0, 0, 1, 0);
    static cv::Rect RotationROI     = cv::Rect(0, 0, 3, 3);
    static cv::Rect TranslationROI  = cv::Rect(3, 0, 1, 3);
    static const double min_percent = 0.75;
    static const double min_error   = 10.0;

    cv::Mat P1 = NoProjection.clone();
    int num_points = args.undistortedPts1.size();

    for (int i = 0; i < args.rotations.size(); ++i)
    {
        for (int j = 0; j < args.translations.size(); ++j)
        {
            cv::Mat rotation    = args.rotations[i];
            cv::Mat translation = args.translations[i];

            cv::Mat P2 = cv::Mat::zeros(3, 4, P1.type());
            rotation.copyTo(P2(RotationROI));
            translation.copyTo(P2(TranslationROI));

            cv::Mat pts_homog_3d, pts_3d;
            cv::triangulatePoints(P1, P2, args.undistortedPts1, args.undistortedPts2, pts_homog_3d);
            cv::convertPointsFromHomogeneous(pts_homog_3d.t(), pts_3d);

            std::vector<unsigned char> status(num_points);
            for (int k = 0; k < num_points; ++k)
                status[k] = (pts_3d.at<double>(k, 2) > 0.0);

            int num_in_front  = cv::countNonZero(status);
            int front_percent = ((double)(num_in_front)) / std::max(1, num_points);

            // Not enough points in front
            if (front_percent < min_percent) continue;

            cv::Vec3d rvec(0, 0, 0);
            cv::Vec3d tvec(0, 0, 0);
            std::vector<cv::Point2d> reprojected_pts1;
            cv::projectPoints(pts_3d, rvec, tvec, in.camera.matrix(), in.camera.distortion(), reprojected_pts1);

            std::vector<cv::Point2d> original_pts1;
            Util::toPoints(args.inliers1, original_pts1);

            const double error_total = cv::norm(cv::Mat(reprojected_pts1), cv::Mat(original_pts1), cv::NORM_L2); 
            const double error_avg   = error_total / std::max((int)reprojected_pts1.size(), 1);

            if (error_avg > min_error) continue;

            for (int k = 0; k < num_points; ++k)
            {
                const double error = cv::norm(original_pts1[k] - reprojected_pts1[k]);
                status[k] &= (error < 2 * min_error);
            }

            out.rotation               = rotation;
            out.translation            = translation;
            out.visible_percent        = front_percent;
            out.avg_reprojection_error = error_avg;

            out.points.clear();
            const cv::Point3d* points_3d = pts_3d.ptr<cv::Point3d>();
            for (int k = 0; k < num_points; ++k)
            {
                if (status[k])
                {
                    out.points.push_back(CloudPoint());
                    out.points[k].pt = cv::Point3d(points_3d[k]);
                    out.points[k].index = args.indices[k];
                }
            }

            return true;
        }
    }

    return false;
}
