
#ifndef __CameraData_HPP__
#define __CameraData_HPP__

#include <core.hpp>

class CameraData
{
private:
    cv::Mat  _matrix;
    cv::Mat  _distortion;
    cv::Size _resolution;

public:
    CameraData(const std::string& calibration_data_path);
    CameraData(cv::Mat camera_matrix, cv::Mat distortion);

    cv::Mat      matrix() const;
    cv::Mat  distortion() const;
    cv::Size resolution() const;

    void resize(const cv::Size& new_res);
};

#endif
