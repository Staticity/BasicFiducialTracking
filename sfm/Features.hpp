
#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__

#include <vector>

namespace cv
{
    class Mat;
    class KeyPoint;
    struct DMatch;
}

namespace Features
{
    void findMatches(
        const cv::Mat& im1,
        const cv::Mat& im2,
        std::vector<cv::KeyPoint>& kp1,
        std::vector<cv::KeyPoint>& kp2,
        cv::Mat& desc1,
        cv::Mat& desc2,
        std::vector<cv::DMatch>& matches);
}

#endif