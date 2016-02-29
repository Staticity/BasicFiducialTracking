#include "Camera.hpp"

#include <string>
#include <assert.h>

Camera::Camera(const std::string& calibration_data_path)
{
    cv::FileStorage fs(calibration_data_path, cv::FileStorage::READ);

    int camera_width, camera_height;
    fs["image_width"]             >> camera_width;
    fs["image_height"]            >> camera_height;
    fs["camera_matrix"]           >> this->_matrix;
    fs["distortion_coefficients"] >> this->_distortion;

    _inverse = _matrix.inv();
    
    fs.release();

    assert(camera_width != 0 && camera_height != 0);

    this->_resolution = cv::Size(camera_width, camera_height);
}

Camera::Camera(cv::Mat camera_matrix, cv::Mat distortion)
: _matrix(camera_matrix)
, _distortion(distortion)
{}

cv::Mat Camera::matrix() const
{
    return _matrix;
}

cv::Mat Camera::inverse() const
{
    return _inverse;
}

cv::Mat Camera::distortion() const
{
    return _distortion;
}

cv::Size Camera::resolution() const
{
    return _resolution;
}

void Camera::resize(const cv::Size& new_res)
{
    assert(new_res.width != 0 && new_res.height != 0);

    double width_ratio = ((double) new_res.width) / _resolution.width;
    double height_ratio = ((double) new_res.height) / _resolution.height;

    _matrix.at<double>(0, 0) *= width_ratio;
    _matrix.at<double>(1, 1) *= height_ratio;
    _matrix.at<double>(0, 2)  = new_res.width * 0.5;
    _matrix.at<double>(1, 2)  = new_res.height * 0.5;

    _resolution = new_res;
    _inverse = _matrix.inv();
}

// Assume projection is identity
cv::Point3d Camera::project(const cv::Point3d p) const
{
    ;
    cv::Mat_<double> projection = _matrix * (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);

    return cv::Point3d(projection(0), projection(1), projection(2));
}

cv::Point3d Camera::normalize(const cv::Point2d p) const
{
    cv::Mat_<double> normalization = _inverse * (cv::Mat_<double>(3, 1) << p.x, p.y, 1.0);

    return cv::Point3d(normalization(0), normalization(1), normalization(2));
}

