#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <vector>
#include <cmath>

namespace Util
{
    template <typename T, typename B>
    void mask(
        const std::vector<T>& v,
        const std::vector<B>& mask,
        std::vector<T>& kept)
    {
        assert(v.size() == mask.size());

        for (int i = 0; i < mask.size(); ++i)
        {
            if (mask[i])
            {
                kept.push_back(v[i]);
            }
        }
    }

    template <typename T>
    bool eq(T a, T b, double eps=10e-9)
    {
        return std::abs(a - b) <= eps;
    }
}

#endif
