
#ifndef __MULTI_VIEW_HPP__
#define __MULTI_VIEW_HPP__

#include <vector>
#include <core.hpp>

class Camera;

namespace cv
{
    template<>
    struct Point_<double>;

    template<>
    struct Point3_<double>;
    
    class Mat;
}

namespace MultiView
{
    void fundamental(
        const std::vector<cv::Point2d>& pts1,
        const std::vector<cv::Point2d>& pts2,
        cv::Mat& F,
        std::vector<unsigned char>& inliers);

    void essential(
        const Camera& c1,
        const Camera& c2,
        const cv::Mat& F,
        cv::Mat& E);

    void getRTfromE(
        const cv::Mat& E,
        std::vector<cv::Mat>& R,
        std::vector<cv::Mat>& T);

    void triangulate(
        const cv::Point2d& x1,
        const cv::Mat_<double>& P1,
        const cv::Point2d& x2,
        const cv::Mat_<double>& P2,
        cv::Point3d& X);

    void triangulate(
        const std::vector<cv::Point2d>& pts1,
        const cv::Mat& P1,
        const std::vector<cv::Point2d>& pts2,
        const cv::Mat& P2,
        std::vector<cv::Point3d>& X);
}

#endif