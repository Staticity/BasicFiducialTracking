#include "CameraData.hpp"

#include <string>
#include <assert.h>

CameraData::CameraData(const std::string& calibration_data_path)
{
    cv::FileStorage fs(calibration_data_path, cv::FileStorage::READ);

    int camera_width, camera_height;
    fs["camera_matrix"] >> this->_matrix;
    fs["distortion_coefficients"] >> this->_distortion;
    fs["image_width"] >> camera_width;
    fs["image_height"] >> camera_height;
    fs.release();

    assert(camera_width != 0 && camera_height != 0);

    this->_resolution = cv::Size(camera_width, camera_height);
}

CameraData::CameraData(cv::Mat camera_matrix, cv::Mat distortion)
: _matrix(camera_matrix)
, _distortion(distortion)
{}

cv::Mat CameraData::matrix() const
{
    return _matrix;
}

cv::Mat CameraData::distortion() const
{
    return _distortion;
}

cv::Size CameraData::resolution() const
{
    return _resolution;
}

void CameraData::resize(const cv::Size& new_res)
{
    assert(new_res.width != 0 && new_res.height != 0);

    double width_ratio = ((double) new_res.width) / _resolution.width;
    double height_ratio = ((double) new_res.height) / _resolution.height;

    _matrix.at<double>(0, 0) *= width_ratio;
    _matrix.at<double>(1, 1) *= height_ratio;
    _matrix.at<double>(0, 2)  = new_res.width * 0.5;
    _matrix.at<double>(1, 2)  = new_res.height * 0.5;

    _resolution = new_res;
}
