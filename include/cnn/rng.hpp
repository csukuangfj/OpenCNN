#pragma once

#include <random>

#include "cnn/Array.hpp"

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
 * Return a random boolean variable according to a bernoulli distribution.
 *
 * @param p the probability to return true
 * @return true with probability p, false with probability 1 - p.
 */
bool bernoulli(double p);

/**
 *
 * Return a guassian distributed random
 * value.
 * @param mean mean of the gaussian
 * @param stddev standard deviation of the gaussian
 */
double gaussian(double mean, double stddev);

/**
 * Fill the array with random values drawn from a gaussian
 * distribution with the given mean and standard deviation.
 *
 * @param arr the array to be filled
 * @param mean mean for the normal distribution
 * @param stddev standard deviation for the normal distribution
 *
 * @note variance is equal to the square of the standard deviation
 */
template<typename Dtype>
void gaussian(Array<Dtype>* arr, Dtype mean, Dtype stddev)
{
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = static_cast<Dtype>(gaussian(mean, stddev));
    }
}

template<typename Dtype>
void uniform(Array<Dtype>* arr, int low, int high)
{
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = static_cast<Dtype>(uniform(low, high));
    }
}

template<typename Dtype>
void bernoulli(Array<Dtype>* arr, double p)
{
    for (int i = 0; i < arr->total_; i++)
    {
        arr->d_[i] = static_cast<Dtype>(bernoulli(p));
    }
}

}  // namespace cnn
