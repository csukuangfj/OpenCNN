#pragma once

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
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = val;
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
template<typename Dtype>
void gaussian(Array<Dtype>* arr, Dtype mean, Dtype stddev);

}  // namespace cnn

