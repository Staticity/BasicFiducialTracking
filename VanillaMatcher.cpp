#include "VanillaMatcher.hpp"

#include <opencv.hpp>
#include <xfeatures2d.hpp>

VanillaMatcher::VanillaMatcher() {}
VanillaMatcher::~VanillaMatcher() {}

bool VanillaMatcher::match(const cv::Mat&             img1,
                           const cv::Mat&             img2,
                           std::vector<cv::KeyPoint>& feat1,
                           std::vector<cv::KeyPoint>& feat2,
                           cv::Mat&                   desc1,
                           cv::Mat&                   desc2,
                           std::vector<cv::DMatch>&   matches) const
{
    bool do_feat1 = feat1.empty();
    bool do_feat2 = feat2.empty();
    bool do_desc1 = desc1.empty();
    bool do_desc2 = desc2.empty();

    cv::Mat gray1, gray2;
    if ((do_feat1 || do_desc1) && img1.channels() != 1)
        cvtColor(img1, gray1, CV_BGR2GRAY);
    else
        gray1 = img1;

    if ((do_feat2 || do_desc2) && img2.channels() != 1)
        cvtColor(img2, gray2, CV_BGR2GRAY);
    else
        gray2 = img2;

    if (do_feat1 || do_feat2)
    {
        // cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create(400);
        // cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(20);
        // cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

        if (do_feat1) detector->detect(img1, feat1);
        if (do_feat2) detector->detect(img2, feat2);

        if (feat1.empty() || feat2.empty())
            return false;
    }

    if (do_desc1 || do_desc2)
    {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SURF::create();
        // cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SIFT::create();
        // cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
        // cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);

        if (do_desc1) extractor->compute(img1, feat1, desc1);
        if (do_desc2) extractor->compute(img2, feat2, desc2);

        if (desc1.empty() || desc2.empty())
            return false;
    }

    // Change distance based on descriptor
    cv::BFMatcher matcher = cv::BFMatcher::BFMatcher(cv::NORM_L2, true);
    matcher.match(desc1, desc2, matches);

    return !matches.empty();
}
