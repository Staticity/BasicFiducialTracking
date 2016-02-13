#include <core.hpp>
#include <vector>
#include <cmath>

class Util
{
public:
    template<typename T>
    static void keepByStatus(const std::vector<T>&                src,
                             const std::vector<unsigned char>& status,
                             std::vector<T>&                      dst);

    static void toPoints(const std::vector<cv::KeyPoint>& keypoints,
                         std::vector<cv::Point>&             points);

    static bool feq(double a, double b, double eps);

};

template<typename T>
void Util::keepByStatus(const std::vector<T>&                src,
                        const std::vector<unsigned char>& status,
                        std::vector<T>&                      dst)
{
    for (int i = 0; i < src.size(); ++i)
        if (status[i])
            dst.push_back(src[i]);
}

void Util::toPoints(const std::vector<cv::KeyPoint>& keypoints,
                    std::vector<cv::Point>&             points)
{
    for (int i = 0; i < keypoints.size(); ++i)
        points.push_back(keypoints[i].pt);
}

static bool Util::feq(double a, double b, double eps=10e-9)
{
    return fabs(a - b) < eps;
}