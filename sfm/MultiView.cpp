#include "MultiView.hpp"

#include <calib3d.hpp>
#include <cmath>

#include "Camera.hpp"
#include "Util.hpp"

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
    const Camera& c1,
    const Camera& c2,
    const cv::Mat& F,
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

    E = K2.t() * F * K1;

    cv::Mat u, w, vt;
    cv::SVD::compute(E, u, w, vt);

    cv::Mat test_rotation  = u * W * vt;
    double rot_determinant = cv::determinant(test_rotation);

    // Want rotation matrix of determinant 1
    if (Util::eq(rot_determinant, -1.0))
    {
        vt.row(2) = -vt.row(2);
    }

    E = u * D * vt;
}

void getRTfromE(
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
    cv::SVD::compute(E, u, w, vt);

    cv::Mat r1 =  u *   W   * vt;
    cv::Mat r2 =  u * W.t() * vt;
    cv::Mat t1 =  u.col(2);
    cv::Mat t2 = -u.col(2);

    R.push_back(r1);
    T.push_back(t1);

    R.push_back(r1);
    T.push_back(t2);

    R.push_back(r2);
    T.push_back(t1);

    R.push_back(r2);
    T.push_back(t2);
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
        A(i, 0) = x1.x * P1(2, i) - P1(0, i);
        A(i, 1) = x1.y * P1(2, i) - P1(1, i);
        A(i, 2) = x2.x * P2(2, i) - P2(0, i);
        A(i, 3) = x2.y * P2(2, i) - P2(1, i);
    }

    cv::Matx43d A2(x1.x * P1(2, 0) - P1(0, 0), x1.x * P1(2, 1) - P1(0, 1), x1.x * P1(2, 2) - P1(0, 2),      
                  x1.y * P1(2, 0) - P1(1, 0), x1.y * P1(2, 1) - P1(1, 1), x1.y * P1(2, 2) - P1(1, 2),      
                  x2.x * P2(2, 0) - P2(0, 0), x2.x * P2(2, 1) - P2(0, 1), x2.x * P2(2, 2) - P2(0, 2),   
                  x2.y * P2(2, 0) - P2(1, 0), x2.y * P2(2, 1) - P2(1, 1), x2.y * P2(2, 2) - P2(1, 2));

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            assert(Util::eq(A(i, j), A2(i, j)));

    cv::Matx41d B(-(x1.x * P1(2, 3) - P1(0, 3)),
                  -(x1.y * P1(2, 3) - P1(1, 3)),
                  -(x2.x * P2(2, 3) - P2(0, 3)),
                  -(x2.y * P2(2, 3) - P2(1, 3)));

    cv::Mat_<double> _X;
    cv::solve(A, B, _X, cv::DECOMP_SVD);
    X = cv::Point3d(_X(0), _X(1), _X(2));
}

void triangulate(
    const std::vector<cv::Point2d>& pts1,
    const cv::Mat& P1,
    const std::vector<cv::Point2d>& pts2,
    const cv::Mat& P2,
    std::vector<cv::Point3d>& X)
{
    assert(pts1.size() == pts2.size());

    const int n = pts1.size();
    X = std::vector<cv::Point3d>(n);
    for (int i = 0; i < n; ++i)
    {
        triangulate(pts1[i], P1, pts2[i], P2, X[i]);
    }
}

