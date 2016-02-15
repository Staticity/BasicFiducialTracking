#include "FlowMatcher.hpp"

#include <opencv.hpp>
#include <features2d.hpp>

#include <set>

#include "Util.hpp"

FlowMatcher::FlowMatcher() {}
FlowMatcher::~FlowMatcher() {}

bool FlowMatcher::match(const cv::Mat&             img1,
                        const cv::Mat&             img2,
                        std::vector<cv::KeyPoint>& feat1,
                        std::vector<cv::KeyPoint>& feat2,
                        cv::Mat&                   desc1,
                        cv::Mat&                   desc2,
                        std::vector<cv::DMatch>&   matches) const
{
    static const float min_of_error = 12.0;
    static const float radius_max   = 2.0;
    static const float max_ratio    = 0.8;

    // Unused
    desc1 = cv::Mat();
    desc2 = cv::Mat();

    // bool do_feat1 = feat1.empty();
    // bool do_feat2 = feat2.empty();
    // bool do_desc1 = desc1.empty();
    // bool do_desc2 = desc2.empty();
    bool do_feat1 = true;
    bool do_feat2 = true;
    bool do_desc1 = true;
    bool do_desc2 = true;

    assert(do_feat1 && do_feat2 && do_desc1 && do_desc2);

    if (do_feat1 || do_feat2)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(20);
        // cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create(400);
        // cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        // cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

        if (do_feat1) detector->detect(img1, feat1);
        if (do_feat2) detector->detect(img2, feat2);

        if (feat1.empty() || feat2.empty())
            return false;
    }

    // No need to use desc1 or desc2

    cv::Mat gray1 = img1;
    cv::Mat gray2 = img2;
    if (img1.channels() != 1) cvtColor(img1, gray1, CV_BGR2GRAY);
    if (img2.channels() != 1) cvtColor(img2, gray2, CV_BGR2GRAY);

    std::vector<cv::Point2f> pts1, pts2;
    Util::toPoints(feat1, pts1);
    Util::toPoints(feat2, pts2);

    std::vector<uchar>       status;
    std::vector<float>       error;
    std::vector<cv::Point2f> moved_pts;
    cv::calcOpticalFlowPyrLK(gray1, gray2, pts1, moved_pts, status, error);


    std::vector<cv::Point2f> inliers;
    std::vector<int>         indices;
    for (int i = 0; i < moved_pts.size(); ++i)
    {
        if (status[i] && error[i] <= min_of_error)
        {
            inliers.push_back(moved_pts[i]);
            indices.push_back(i);
        }
        else
        {
            status[i] = 0;
        }
    }

    cv::Mat flat_inliers = cv::Mat(inliers).reshape(1);
    cv::Mat flat_points2 = cv::Mat(pts2).reshape(1);

    cv::BFMatcher matcher = cv::BFMatcher::BFMatcher(cv::NORM_L2, true);

    std::vector<std::vector<cv::DMatch> > neighbors;
    matcher.radiusMatch(flat_inliers, flat_points2, neighbors, radius_max);

    std::set<int> used;
    for (int i = 0; i < neighbors.size(); ++i)
    {
        bool found = false;
        cv::DMatch best_match;

        switch (neighbors[i].size())
        {
            case 0:
                break;
            case 1:
                best_match = neighbors[i][0];
                found = true;
                break;
            default: // > 2
                double ratio = neighbors[i][0].distance / neighbors[i][1].distance;
                if (ratio < max_ratio)
                {
                    best_match = neighbors[i][0];
                    found = true;
                }
                break;
        }

        // Found and unique
        if (found && used.find(best_match.trainIdx) == used.end())
        {
            // Go back to the original mapping of indices
            best_match.queryIdx = indices[best_match.queryIdx];
            used.insert(best_match.trainIdx);
            matches.push_back(best_match);
        }
    }

    return !matches.empty();
}
