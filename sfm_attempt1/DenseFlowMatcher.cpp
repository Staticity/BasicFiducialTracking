#include "DenseFlowMatcher.hpp"

#include <opencv.hpp>
#include <features2d.hpp>

#include <set>

#include "Util.hpp"

DenseFlowMatcher::DenseFlowMatcher() {}
DenseFlowMatcher::~DenseFlowMatcher() {}

bool DenseFlowMatcher::match(const cv::Mat&             img1,
                        const cv::Mat&             img2,
                        std::vector<cv::KeyPoint>& feat1,
                        std::vector<cv::KeyPoint>& feat2,
                        cv::Mat&                   desc1,
                        cv::Mat&                   desc2,
                        std::vector<cv::DMatch>&   matches) const
{
    static const double pyrmid_scale = 0.5;
    static const int    levels       = 3;
    static const int    window_size  = 15;
    static const int    iterations   = 3;
    static const int    poly_n       = 5;
    static const double poly_sigma   = 1.1;
    static const int    flags        = 0;

    Mat flow;

    calcOpticalFlowFarneback(
        img1,
        img2,
        flow,
        pyrmid_scale,
        levels,
        window_size,
        iteraitons,
        poly_n,
        poly_sigma,
        flags);

    

    return !matches.empty();
}
