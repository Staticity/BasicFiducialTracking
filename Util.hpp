
// #ifndef __Util_HPP__
// #define __Util_HPP__

#include <core.hpp>
#include <vector>
#include <cmath>

namespace Util
{
  template<typename T>
  inline void retain(const std::vector<T>&              src,
                     const std::vector<unsigned char>& mask,
                     std::vector<T>&                    dst)
  {
      for (int i = 0; i < src.size(); ++i)
          if (mask[i])
              dst.push_back(src[i]);
  }

  inline void toPoints(const std::vector<cv::KeyPoint>& keypoints,
                       std::vector<cv::Point2d>&           points)
  {
      for (int i = 0; i < keypoints.size(); ++i)
          points.push_back(keypoints[i].pt);
  }

  inline bool feq(double a, double b, double eps=10e-9)
  {
      return fabs(a - b) < eps;
  }

};

// #endif
