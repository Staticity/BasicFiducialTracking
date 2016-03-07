
#ifndef __MULTI_VIEW_HPP__
#define __MULTI_VIEW_HPP__

#include <vector>
#include <core.hpp>

class Camera;

namespace MultiView
{
    void fundamental(
        const std::vector<cv::Point2d>& pts1,
        const std::vector<cv::Point2d>& pts2,
        cv::Mat& F,
        std::vector<unsigned char>& inliers);

    void essential(
        const cv::Mat& F,
        const Camera& c1,
        const Camera& c2,
        cv::Mat& E);

    void get_rotation_and_translation(
        const cv::Mat& E,
        std::vector<cv::Mat>& R,
        std::vector<cv::Mat>& T);

    void get_projection(
        const cv::Mat& R,
        const cv::Mat& T,
        cv::Mat& P);

    void transform(
        const cv::Point3d& point,
        const cv::Mat_<double>& rotation,
        const cv::Mat_<double>& translation,
        cv::Point3d& transformed_point);

    void triangulate(
        const cv::Point2d& x1,
        const cv::Mat_<double>& P1,
        const cv::Point2d& x2,
        const cv::Mat_<double>& P2,
        cv::Point3d& point);
    
    void iterative_triangulate(
        const cv::Point3d& u,
        const cv::Matx34d& P1,
        const cv::Point3d& v,
        const cv::Matx34d& P2,
        cv::Point3d& point);

    void triangulate(
        const std::vector<cv::Point2d>& pts1,
        const cv::Mat& P1,
        const cv::Mat& K1,
        const std::vector<cv::Point2d>& pts2,
        const cv::Mat& P2,
        const cv::Mat& K2,
        std::vector<cv::Point3d>& points);

    void triangulate(
        const std::vector<cv::Point2d>& pts1,
        const cv::Mat& K1,
        const std::vector<cv::Point2d>& pts2,
        const cv::Mat& K2,
        const cv::Mat& rotation,
        const cv::Mat& translation,
        std::vector<cv::Point3d>& points);

    void project(
        const cv::Point3d& point,
        const cv::Mat& rotation,
        const cv::Mat& translation,
        const cv::Mat& camera,
        cv::Point2d& coordinate);

    void project(
        const cv::Point3d& point,
        const cv::Mat& projection,
        cv::Point2d& coordinate);
    
    void project(
        const std::vector<cv::Point3d>& points,
        const cv::Mat& rotation,
        const cv::Mat& translation,
        const cv::Mat& camera,
        std::vector<cv::Point2d>& coordinates);

    void project(
        const std::vector<cv::Point3d>& points,
        const cv::Mat& projection,
        std::vector<cv::Point2d>& coordinates);
}

#endif