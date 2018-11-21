#pragma once

#include <random>

#include "cnn/array.hpp"

namespace cnn
{

/**
 * Set the seed for the random number generator.
 * @param val the value of the seed.
 */
void set_seed(int val);

/**
 * Return a uniformly distributed random value in the given
 * interval. Both ends are inclusive.
 *
 * @param low the lower bound, inclusive
 * @param high the upper bound, inclusive
 * @return a random value uniformly distributed in the interval [low, high].
 */
int uniform(int low, int high);

/**
 *
 * compute
 *  (alpha*x[0]-beta*y[0])**2 + (alpha*x[1]-beta*y[1])**2
 *  + ... + (alpha*x[n-1]-beta*y[n-1])**2
 *
 *  @param n number of elements in x and y
 *  @param alpha every element in x is scaled by alpha
 *  @param x an array of n elements
 *  @param beta every element in y is scaled by beta
 *  @param y an array of n elements
 *  @return \f[\sum_{i=0}^{n-1} (\alpha x[i] - \beta y[i])^2 \f]
 */
template<typename Dtype>
Dtype ax_sub_by_squared(int n, Dtype alpha, const Dtype* x,
        Dtype beta, const Dtype* y)
{
    Dtype res = 0;
    for (int i = 0; i < n; i++)
    {
        Dtype diff = alpha*x[i] - beta*y[i];
        res += diff*diff;
    }
    return res;
}

/**
 * Compute the dot product between alpha*x and beta*y;
 * alpha*x[0]*beta*y[0] + ... + alpha*x[n-1]*beta*y[n-1]
 *
 * @param n number of elements in x and y
 * @param alpha every element in x is scaled by alpha
 * @param x an array of n elements
 * @param beta every element in y is scaled by beta
 * @param y an array of n elements
 * @return \f[ sum_{i=0}^{n-1} (\alpha x[i] + \beta y[i]) \f]
 */
template<typename Dtype>
Dtype ax_dot_by(int n, Dtype alpha, const Dtype* x,
        Dtype beta, const Dtype* y)
{
    Dtype res = 0;
    for (int i = 0; i < n; i++)
    {
        res += alpha * x[i] * beta * y[i];
    }
    return res;
}

/**
 * y[i] = alpha*x[i] + beta*y[i]
 * @param n number of elements
 * @param alpha every element in x is scaled by alpha
 * @param x an array of n elements
 * @param beta every element in y is scaled by beta
 * @param y an array of n elements
 * @return void
 */
template<typename Dtype>
void ax_plus_by(int n, Dtype alpha, const Dtype* x,
        Dtype beta, Dtype* y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

/**
 * Set all elements of the array to the specified value.
 * After this operation,
 * arr[0]=arr[1]=...=arr[total]=val
 *
 * @param arr the array to be set
 * @param val the given value
 */
template<typename Dtype>
void set_to(Array<Dtype>* arr, Dtype val)
{
    if (!val)
    {
        memset(arr->d_, 0, sizeof(Dtype)*arr->total_);
    }
    else    // NOLINT
    {
        for (int i = 0; i < arr->total_; i++)
        {
            arr->d_[i] = val;
        }
    }
}

/**
 * dst[i] = alpha * src[i]
 */
template<typename Dtype>
void scale_arr(Dtype alpha, const Array<Dtype>& src, Array<Dtype> *dst)
{
    for (int i = 0; i < src.total_; i++)
    {
        dst->d_[i] = alpha * src.d_[i];
    }
}

/**
 * res = src[0] + src[1] + src[2] + ... + src[total-1]
 */
template<typename Dtype>
Dtype sum_arr(const Array<Dtype>& src)
{
    Dtype res = 0;
    for (int i = 0; i < src.total_; i++)
    {
        res += src.d_[i];
    }
    return res;
}

template<typename Dtype>
Dtype sum_arr(int n, const Dtype* src)
{
    Dtype res = 0;
    for (int i = 0; i < n; i++)
    {
        res += src[i];
    }
    return res;
}

/**
 * dst[i] = src[i] - alpha
 */
template<typename Dtype>
void sub_scalar(Dtype alpha, const Array<Dtype>& src, Array<Dtype> *dst)
{
    for (int i = 0; i < src.total_; i++)
    {
        dst->d_[i] = src.d_[i] - alpha;
    }
}

/**
 * dst[i] = src[i] - alpha
 */
template<typename Dtype>
void sub_scalar(int n, Dtype alpha, const Dtype* src, Dtype* dst)
{
    for (int i = 0; i < n; i++)
    {
        dst[i] = src[i] - alpha;
    }
}

/**
 * Fill the array with random values drawn from a guassian
 * distribution with the given mean and standard deviation.
 *
 * @param arr the array to be filled
 * @param mean mean for the normal distribution
 * @param stddev standard deviation for the normal distribution
 *
 * @note variance is equal to the square of the standard deviation
 */

// refer to
// http://www.cplusplus.com/reference/random/normal_distribution/normal_distirbution/
template<typename Dtype>
void gaussian(Array<Dtype>* arr, Dtype mean, Dtype stddev)
{
    extern std::default_random_engine g_generator;
    // we use double instead of Dtype here because it causes
    // error on Linux when Dtype is Jet<>
    std::normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = distribution(g_generator);
    }
}

template<typename Dtype>
void uniform(Array<Dtype>* arr, int low, int high)
{
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = uniform(low, high);
    }
}


}  // namespace cnn

#include "../../src/array_math.cpp"

