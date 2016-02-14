
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
        std::vector<uchar> mask;
        std::vector<int>           indices;
        // std::vector<cv::KeyPoint>  inliers1;
        // std::vector<cv::KeyPoint>  inliers2;
        std::vector<cv::Point2d>   inliers1;
        std::vector<cv::Point2d>   inliers2;
        std::vector<cv::Mat>       rotations;
        std::vector<cv::Mat>       translations;
        cv::Mat                    undistortedPts1;
        cv::Mat                    undistortedPts2;
    };

    bool _computeFundamental(const Input& in, Args& args, Tracker::Output& out) const;

    bool   _computeEssential(const Input& in, Args& args, Tracker::Output& out) const;

    bool    _undistortPoints(const Input& in, Args& args, Tracker::Output& out) const;

    bool        _triangulate(const Input& in, Args& args, Tracker::Output& out) const;

    bool           _getCloud(const Input& in, Args& args, Tracker::Output& out) const;

    cv::Mat_<double> _iterativeTriangulate(const cv::Point3d& u, const cv::Matx34d& P1,
                                           const cv::Point3d& v, const cv::Matx34d& P2) const;

    cv::Mat_<double>     _triangulatePoint(const cv::Point3d& u, const cv::Matx34d& P1,
                                           const cv::Point3d& v, const cv::Matx34d& P2) const;
};

#endif
