
#ifndef __Tracker_HPP__
#define __Tracker_HPP__

#include <core.hpp>
#include <vector>

#include "CameraData.hpp"

class Tracker
{
public:

    Tracker();
    virtual ~Tracker();

    struct CloudPoint
    {
        cv::Point3f pt;
        int index;
    };

    struct Input
    {
        const std::vector<cv::KeyPoint> feat1;
        const std::vector<cv::KeyPoint> feat2;
        const CameraData               camera;
    };

    struct Output
    {
        std::vector<unsigned char> mask;
        
        cv::Mat                    fundamental;
        cv::Mat                    essential;
        cv::Mat                    essentialU;
        cv::Mat                    essentialW;
        cv::Mat                    essentialVt;
        
        cv::Mat                    rotation;
        cv::Mat                    translation;
        std::vector<CloudPoint>    points;

        float                      visible_percent;
        float                      avg_reprojection_error;
    };

    virtual bool triangulate(const Input& in, Output& out) const = 0;

};

#endif
