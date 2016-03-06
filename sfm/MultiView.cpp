#include "MultiView.hpp"

#include <calib3d.hpp>
#include <cmath>
#include <iostream>

#include "Camera.hpp"
#include "Util.hpp"

namespace MultiView
{
    void fundamental(
        const std::vector<cv::Point2d>& pts1,
        const std::vector<cv::Point2d>& pts2,
        cv::Mat& F,
        std::vector<unsigned char>& inliers)
    {
        double maxV1, maxV2;
        cv::minMaxIdx(pts1, 0, &maxV1);
        cv::minMaxIdx(pts2, 0, &maxV2);

        const double maxV = std::max(maxV2, maxV2);
        const double magic      = 0.66;
        const double confidence = 0.99;
        F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, magic * maxV, confidence, inliers);
    }

    void essential(
        const cv::Mat& F,
        const Camera& c1,
        const Camera& c2,
        cv::Mat& E)
    {
        static cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1,  0,
                                                      1,  0,  0,
                                                      0,  0,  1);

        static cv::Mat D = (cv::Mat_<double>(3, 3) << 1,  0,  0,
                                                      0,  1,  0,
                                                      0,  0,  0);

        const cv::Mat K1 = c1.matrix();
        const cv::Mat K2 = c2.matrix();

        // K2.rows because K2.t()
        assert(K2.rows == F.rows && F.cols == K1.rows);
        E = K2.t() * F * K1;

        cv::Mat u, w, vt;
        cv::SVD::compute(E, w, u, vt);

        assert(u.cols == W.rows && W.cols == vt.rows);
        cv::Mat test_rotation  = u * W * vt;
        double rot_determinant = cv::determinant(test_rotation);

        // Want rotation matrix of determinant 1
        if (Util::eq(rot_determinant, -1.0))
        {
            vt.row(2) = -vt.row(2);
        }

        assert(u.cols == D.rows && D.cols == vt.rows);
        E = u * D * vt;
    }

    void get_rotation_and_translation(
        const cv::Mat& E,
        std::vector<cv::Mat>& R,
        std::vector<cv::Mat>& T)
    {        
        static cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1,  0,
                                                      1,  0,  0,
                                                      0,  0,  1);

        static cv::Mat D = (cv::Mat_<double>(3, 3) << 1,  0,  0,
                                                      0,  1,  0,
                                                      0,  0,  0);
        cv::Mat u, w, vt;
        cv::SVD::compute(E, w, u, vt);

        assert(u.cols == W.rows && W.rows == vt.cols);
        assert(u.cols == W.cols && W.rows == vt.cols);

        cv::Mat r1 =  u *   W   * vt;
        cv::Mat r2 =  u * W.t() * vt;
        cv::Mat t1 =  u.col(2);
        cv::Mat t2 = -u.col(2);

        assert(r1.size() == cv::Size(3, 3));
        assert(r2.size() == cv::Size(3, 3));
        assert(t1.size() == cv::Size(1, 3));
        assert(t2.size() == cv::Size(1, 3));

        R.push_back(r1);
        T.push_back(t1);

        R.push_back(r1);
        T.push_back(t2);

        R.push_back(r2);
        T.push_back(t1);

        R.push_back(r2);
        T.push_back(t2);
    }

    void get_projection(
        const cv::Mat& R,
        const cv::Mat& T,
        cv::Mat& P)
    {
        P = cv::Mat::zeros(3, 4, CV_64F);
        R.copyTo(P(cv::Rect(0, 0, 3, 3)));
        T.copyTo(P(cv::Rect(3, 0, 1, 3)));
    }

    void triangulate(
        const cv::Point2d& x1,
        const cv::Mat_<double>& P1,
        const cv::Point2d& x2,
        const cv::Mat_<double>& P2,
        cv::Point3d& X)
    {
        cv::Matx43d A;
        for (int i = 0; i <= 2; ++i)
        {
            A(0, i) = x1.x * P1(2, i) - P1(0, i);
            A(1, i) = x1.y * P1(2, i) - P1(1, i);
            A(2, i) = x2.x * P2(2, i) - P2(0, i);
            A(3, i) = x2.y * P2(2, i) - P2(1, i);
        }

        cv::Matx41d B(-(x1.x * P1(2, 3) - P1(0, 3)),
                      -(x1.y * P1(2, 3) - P1(1, 3)),
                      -(x2.x * P2(2, 3) - P2(0, 3)),
                      -(x2.y * P2(2, 3) - P2(1, 3)));

        // cv::Matx43d A(x1.x * P1(2, 0) - P1(0, 0), x1.x * P1(2, 1) - P1(0, 1), x1.x * P1(2, 2) - P1(0, 2),      
        //               x1.y * P1(2, 0) - P1(1, 0), x1.y * P1(2, 1) - P1(1, 1), x1.y * P1(2, 2) - P1(1, 2),      
        //               x2.x * P2(2, 0) - P2(0, 0), x2.x * P2(2, 1) - P2(0, 1), x2.x * P2(2, 2) - P2(0, 2),   
        //               x2.y * P2(2, 0) - P2(1, 0), x2.y * P2(2, 1) - P2(1, 1), x2.y * P2(2, 2) - P2(1, 2));

        cv::Mat_<double> _X;
        cv::solve(A, B, _X, cv::DECOMP_SVD);
        X = cv::Point3d(_X(0), _X(1), _X(2));
    }

    void triangulate(
        const std::vector<cv::Point2d>& pts1,
        const std::vector<cv::Point2d>& pts2,
        const cv::Mat& rotation,
        const cv::Mat& translation,
        std::vector<cv::Point3d>& points)
    {
        assert(rotation.size() == cv::Size(3, 3));
        assert(translation.size() == cv::Size(1, 3));

        cv::Mat no_rotation = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat no_translation = cv::Mat::zeros(3, 1, CV_64F);

        cv::Mat P1, P2;
        get_projection(no_rotation, no_translation, P1);
        get_projection(rotation, translation, P2);

        triangulate(pts1, P1, pts2, P2, points);
    }

    void triangulate(
        const std::vector<cv::Point2d>& pts1,
        const cv::Mat& P1,
        const std::vector<cv::Point2d>& pts2,
        const cv::Mat& P2,
        std::vector<cv::Point3d>& points)
    {
        assert(P1.size() == cv::Size(4, 3));
        assert(P2.size() == cv::Size(4, 3));
        assert(P1.type() == CV_64F);
        assert(P2.type() == CV_64F);
        assert(pts1.size() == pts2.size());

        const int n = pts1.size();
        points = std::vector<cv::Point3d>(n);
        for (int i = 0; i < n; ++i)
        {
            triangulate(pts1[i], P1, pts2[i], P2, points[i]);
        }
    }

    void project(
        const cv::Point3d& point,
        const cv::Mat& rotation,
        const cv::Mat& translation,
        cv::Point2d& coordinate)
    {
        cv::Mat projection;
        get_projection(rotation, translation, projection);

        project(point, projection, coordinate);
    }

    void project(
        const cv::Point3d& point,
        const cv::Mat& projection,
        cv::Point2d& coordinate)
    {
        assert(projection.size() == cv::Size(4, 3));

        cv::Mat pt = (cv::Mat_<double>(4, 1) << point.x,
                                                point.y,
                                                point.z,
                                                1.0);
        cv::Mat proj_pt = projection * pt;

        double wx = proj_pt.at<double>(0);
        double wy = proj_pt.at<double>(1);
        double w = proj_pt.at<double>(2);

        assert(w != 0.0);
        double x = wx / w;
        double y = wy / y;

        coordinate = cv::Point2d(x, y);
    }

    void project(
        const std::vector<cv::Point3d>& points,
        const cv::Mat& rotation,
        const cv::Mat& translation,
        std::vector<cv::Point2d>& coordinates)
    {
        cv::Mat projection;
        get_projection(rotation, translation, projection);

        project(points, projection, coordinates);
    }

    void project(
        const std::vector<cv::Point3d>& points,
        const cv::Mat& projection,
        std::vector<cv::Point2d>& coordinates)
    {
        for (int i = 0; i < points.size(); ++i)
        {
            project(points[i], projection, coordinates[i]);
        }
    }
}
