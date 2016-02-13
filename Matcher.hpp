
#ifndef __Matcher_HPP__
#define __Matcher_HPP__

#include <vector>

namespace cv
{
    class Mat;
    struct KeyPoint;
    struct DMatch;
}

class Matcher
{
public:
    Matcher();
    virtual ~Matcher();
           
    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::DMatch>&   matches) const;

    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::KeyPoint>& feat1,
                       std::vector<cv::KeyPoint>& feat2,
                       std::vector<cv::DMatch>&   matches) const;

    virtual bool match(const cv::Mat&             img1,
                       const cv::Mat&             img2,
                       std::vector<cv::KeyPoint>& feat1,
                       std::vector<cv::KeyPoint>& feat2,
                       cv::Mat&                   desc1,
                       cv::Mat&                   desc2,
                       std::vector<cv::DMatch>&   matches) const = 0;
};

#endif
