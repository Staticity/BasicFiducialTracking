
// #ifndef __Util_HPP__
// #define __Util_HPP__

#include <core.hpp>
#include <vector>
#include <cmath>

namespace Util
{
    template<typename T, typename B>
    inline void retain(const std::vector<T>&              src,
                       const std::vector<B>&             mask,
                       std::vector<T>&                    dst)
    {
        for (int i = 0; i < src.size(); ++i)
            if (mask[i])
                dst.push_back(src[i]);
    }

    inline void toPoints(const std::vector<cv::KeyPoint>&   feat1,
                         const std::vector<cv::KeyPoint>&   feat2,
                         const std::vector<cv::DMatch>&     matches,
                         std::vector<cv::Point2d>&          pts1,
                         std::vector<cv::Point2d>&          pts2)
    {
        for (int i = 0; i < matches.size(); ++i)
        {
            pts1.push_back(cv::Point2d(feat1[matches[i].queryIdx].pt));
            pts2.push_back(cv::Point2d(feat2[matches[i].trainIdx].pt));
        }
    }

    inline void toPoints(const std::vector<cv::KeyPoint>& keypoints,
                         std::vector<cv::Point2d>&           points)
    {
        for (int i = 0; i < keypoints.size(); ++i)
            points.push_back(cv::Point2d(keypoints[i].pt));
    }

    inline bool feq(double a, double b, double eps=10e-9)
    {
        return fabs(a - b) < eps;
    }

};

// #endif
