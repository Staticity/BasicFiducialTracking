
#ifndef __DenseFlowMatcher_HPP__
#define __DenseFlowMatcher_HPP__

#include "Matcher.hpp"

class DenseFlowMatcher : public Matcher
{
public:
    DenseFlowMatcher();
    virtual ~DenseFlowMatcher();

    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::KeyPoint>& feat1,
                       std::vector<cv::KeyPoint>& feat2,
                       cv::Mat&                   desc1,
                       cv::Mat&                   desc2,
                       std::vector<cv::DMatch>&   matches) const;
};

#endif
