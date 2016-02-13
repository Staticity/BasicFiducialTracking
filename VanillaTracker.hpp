
#ifndef __VanillaTracker_HPP__
#define __VanillaTracker_HPP__

#include "Tracker.hpp"

class VanillaTracker : public Tracker
{
public:

    VanillaTracker();
    virtual ~VanillaTracker();

    virtual bool triangulate(const Input& in, Output& out) const;

private:

    struct Args
    {
        std::vector<unsigned char> mask;
        std::vector<int>           indices;
        std::vector<cv::KeyPoint>  inliers1;
        std::vector<cv::KeyPoint>  inliers2;
        std::vector<cv::Mat>       rotations;
        std::vector<cv::Mat>       translations;
        std::vector<cv::Point2d>   undistortedPts1;
        std::vector<cv::Point2d>   undistortedPts2;
    };

    bool _computeFundamental(const Input& in, Args& args, Tracker::Output& out) const;

    bool _computeEssential(const Input& in, Args& args, Tracker::Output& out) const;

    bool _undistortPoints(const Input& in, Args& args, Tracker::Output& out) const;

    bool _triangulate(const Input& in, Args& args, Tracker::Output& out) const;

};

#endif
