#pragma once

#include "cnn/array.hpp"

namespace cnn
{

/**
 *
 * compute
 *  (alpha*x[0]-beta*y[0])**2 + (alpha*x[1]-beta*y[1])**2
 *  + ... + (alpha*x[n-1]-beta*y[n-1])**2
 */
template<typename Dtype>
Dtype ax_sub_by_squared(int n, Dtype alpha, Dtype* x, Dtype beta, Dtype* y)
{
    Dtype res = 0;
    for (int i = 0; i < n; i++)
    {
        Dtype diff = alpha*x[i] - beta*y[i];
        res += diff*diff;
    }
    return res;
}


template<typename Dtype>
void set_to(Array<Dtype>* arr, Dtype val)
{
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = val;
    }
}

}  // namespace cnn
