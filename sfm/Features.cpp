#include "Features.hpp"

#include <core.hpp>
#include <xfeatures2d.hpp>

namespace Features
{
    void findMatches(
        const cv::Mat& im1,
        const cv::Mat& im2,
        std::vector<cv::KeyPoint>& kp1,
        std::vector<cv::KeyPoint>& kp2,
        cv::Mat& desc1,
        cv::Mat& desc2,
        std::vector<cv::DMatch>& matches)
    {        
        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
        akaze->detectAndCompute(im1, cv::noArray(), kp1, desc1);
        akaze->detectAndCompute(im2, cv::noArray(), kp2, desc2);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch> > nn_matches;
        matcher.knnMatch(desc1, desc2, nn_matches, 2);

        const double ratio_test_thresh = 0.8f;
        for (int i = 0; i < nn_matches.size(); ++i)
        {
            assert(nn_matches[i].size() == 2);

            const cv::DMatch& first = nn_matches[i][0];
            float dist1 = nn_matches[i][0].distance;
            float dist2 = nn_matches[i][1].distance;

            if (dist1 < ratio_test_thresh * dist2)
            {
                matches.push_back(first);
            }
        }
    }
}
