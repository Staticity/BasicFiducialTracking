
#ifndef __VanillaMatcher_HPP__
#define __VanillaMatcher_HPP__

#include "Matcher.hpp"

class VanillaMatcher : public Matcher
{
public:
    VanillaMatcher();
    virtual ~VanillaMatcher();

    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::KeyPoint>& feat1,
                       std::vector<cv::KeyPoint>& feat2,
                       cv::Mat&                   desc1,
                       cv::Mat&                   desc2,
                       std::vector<cv::DMatch>&   matches) const;
};

#endif
