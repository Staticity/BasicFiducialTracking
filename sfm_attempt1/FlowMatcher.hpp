
#ifndef __FlowMatcher_HPP__
#define __FlowMatcher_HPP__

#include "Matcher.hpp"

class FlowMatcher : public Matcher
{
public:
    FlowMatcher();
    virtual ~FlowMatcher();

    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::KeyPoint>& feat1,
                       std::vector<cv::KeyPoint>& feat2,
                       cv::Mat&                   desc1,
                       cv::Mat&                   desc2,
                       std::vector<cv::DMatch>&   matches) const;
};

#endif
