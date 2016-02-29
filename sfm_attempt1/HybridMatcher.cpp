#include "HybridMatcher.hpp"

#include <opencv.hpp>
#include <xfeatures2d.hpp>

#include "VanillaMatcher.hpp"
#include "FlowMatcher.hpp"

HybridMatcher::HybridMatcher() {}
HybridMatcher::~HybridMatcher() {}

bool HybridMatcher::match(const cv::Mat&             img1,
                          const cv::Mat&             img2,
                          std::vector<cv::KeyPoint>& feat1,
                          std::vector<cv::KeyPoint>& feat2,
                          cv::Mat&                   desc1,
                          cv::Mat&                   desc2,
                          std::vector<cv::DMatch>&   matches) const
{
    // Reset user input
    feat1.clear();
    feat2.clear();
    matches.clear();
    assert(feat1.empty());
    assert(feat2.empty());
    assert(matches.empty());

    // Use standard feature matching along with flow
    VanillaMatcher vm;
    FlowMatcher fm;

    std::vector<cv::KeyPoint> vm_f1, vm_f2, fm_f1, fm_f2;
    cv::Mat                   vm_d1, vm_d2, fm_d1, fm_d2;
    std::vector<cv::DMatch>   vm_matches, fm_matches;

    vm.match(img1, img2, vm_f1, vm_f2, vm_d1, vm_d2, vm_matches);
    fm.match(img1, img2, fm_f1, fm_f2, fm_d1, fm_d2, fm_matches);

    feat1.insert(feat1.end(), vm_f1.begin(), vm_f1.end());
    feat1.insert(feat1.end(), fm_f1.begin(), fm_f1.end());

    feat2.insert(feat2.end(), vm_f2.begin(), vm_f2.end());
    feat2.insert(feat2.end(), fm_f2.begin(), fm_f2.end());

    matches.insert(matches.end(), vm_matches.begin(), vm_matches.end());
    matches.insert(matches.end(), fm_matches.begin(), fm_matches.end());

    // Update the fm_matches
    for (int i = vm_matches.size(); i < matches.size(); ++i)
    {
        matches[i].queryIdx += vm_f1.size();
        matches[i].trainIdx += vm_f2.size();
    }

    return !matches.empty();
}
