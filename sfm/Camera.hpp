
#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include <core.hpp>

class Camera
{
private:
    cv::Mat  _matrix;
    cv::Mat  _inverse;
    cv::Mat  _distortion;
    cv::Size _resolution;

public:
    Camera(const std::string& calibration_data_path);
    Camera(cv::Mat camera_matrix, cv::Mat distortion);

    cv::Mat      matrix() const;
    cv::Mat     inverse() const;
    cv::Mat  distortion() const;
    cv::Size resolution() const;

    void resize(const cv::Size& new_res);

    cv::Point3d project(const cv::Point3d p) const;
    cv::Point3d normalize(const cv::Point2d p) const;
};

#endif
