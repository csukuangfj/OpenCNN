/*  ---------------------------------------------------------------------
  Copyright 2018-2019 Fangjun Kuang
  email: csukuangfj at gmail dot com
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a COPYING file of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>
  -----------------------------------------------------------------  */
#ifndef CNN_UTILS_RNG_H_
#define CNN_UTILS_RNG_H_

#include <random>

#include "cnn/array/array.h"

namespace cnn {

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
template <typename Dtype>
void gaussian(Array<Dtype>* arr, Dtype mean, Dtype stddev) {
  for (int i = 0; i < arr->total_; i++) {
    arr->d_[i] = static_cast<Dtype>(gaussian(mean, stddev));
  }
}

template <typename Dtype>
void uniform(Array<Dtype>* arr, int low, int high) {
  for (int i = 0; i < arr->total_; i++) {
    arr->d_[i] = static_cast<Dtype>(uniform(low, high));
  }
}

template <typename Dtype>
void bernoulli(Array<Dtype>* arr, double p) {
  for (int i = 0; i < arr->total_; i++) {
    arr->d_[i] = static_cast<Dtype>(bernoulli(p));
  }
}

}  // namespace cnn
#endif  // CNN_UTILS_RNG_H_
