
#ifndef __HybridMatcher_HPP__
#define __HybridMatcher_HPP__

#include "Matcher.hpp"

class HybridMatcher : public Matcher
{
public:
    HybridMatcher();
    virtual ~HybridMatcher();

    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::KeyPoint>& feat1,
                       std::vector<cv::KeyPoint>& feat2,
                       cv::Mat&                   desc1,
                       cv::Mat&                   desc2,
                       std::vector<cv::DMatch>&   matches) const;
};

#endif
