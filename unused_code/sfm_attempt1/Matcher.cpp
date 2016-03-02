#include "Matcher.hpp"

#include <core.hpp>

Matcher::Matcher() {}
Matcher::~Matcher() {}

bool Matcher::match(const cv::Mat&             img1,
                    const cv::Mat&             img2,
                    std::vector<cv::DMatch>&   matches) const
{
    std::vector<cv::KeyPoint> feat1, feat2;
    return match(img1, img2, feat1, feat2, matches);
}

bool Matcher::match(const cv::Mat&             img1,
                    const cv::Mat&             img2,
                    std::vector<cv::KeyPoint>& feat1,
                    std::vector<cv::KeyPoint>& feat2,
                    std::vector<cv::DMatch>&   matches) const
{
    cv::Mat desc1, desc2;
    return match(img1, img2, feat1, feat2, desc1, desc2, matches);
}
