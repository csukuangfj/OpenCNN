#pragma once

#include "cnn/array.hpp"

namespace cnn {

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
template <typename Dtype>
Dtype ax_sub_by_squared(int n, Dtype alpha, const Dtype* x, Dtype beta,
                        const Dtype* y) {
  Dtype res = 0;
  for (int i = 0; i < n; i++) {
    Dtype diff = alpha * x[i] - beta * y[i];
    res += diff * diff;
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
template <typename Dtype>
Dtype ax_dot_by(int n, Dtype alpha, const Dtype* x, Dtype beta,
                const Dtype* y) {
  Dtype res = 0;
  for (int i = 0; i < n; i++) {
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
template <typename Dtype>
void ax_plus_by(int n, Dtype alpha, const Dtype* x, Dtype beta, Dtype* y) {
  for (int i = 0; i < n; i++) {
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
template <typename Dtype>
void set_to(Array<Dtype>* arr, Dtype val) {
  if (!val) {
    memset(arr->d_, 0, sizeof(Dtype) * arr->total_);
  } else  // NOLINT
  {
    for (int i = 0; i < arr->total_; i++) {
      arr->d_[i] = val;
    }
  }
}

template <typename Dtype>
void set_to(int n, Dtype* arr, Dtype val) {
  if (!val) {
    memset(arr, 0, sizeof(Dtype) * n);
  } else  // NOLINT
  {
    for (int i = 0; i < n; i++) {
      arr[i] = val;
    }
  }
}

/**
 * dst[i] = alpha * src[i]
 */
template <typename Dtype>
void scale_arr(Dtype alpha, const Array<Dtype>& src, Array<Dtype>* dst) {
  for (int i = 0; i < src.total_; i++) {
    dst->d_[i] = alpha * src.d_[i];
  }
}

template <typename Dtype>
void scale_arr(int n, Dtype alpha, const Dtype* src, Dtype* dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = alpha * src[i];
  }
}

/**
 * res = src[0] + src[1] + src[2] + ... + src[total-1]
 */
template <typename Dtype>
Dtype sum_arr(const Array<Dtype>& src) {
  Dtype res = 0;
  for (int i = 0; i < src.total_; i++) {
    res += src.d_[i];
  }
  return res;
}

template <typename Dtype>
Dtype sum_arr(int n, const Dtype* src) {
  Dtype res = 0;
  for (int i = 0; i < n; i++) {
    res += src[i];
  }
  return res;
}

template <typename Dtype>
Dtype sum_squared_arr(int n, const Dtype* src) {
  Dtype res = 0;
  for (int i = 0; i < n; i++) {
    res += src[i] * src[i];
  }
  return res;
}

/**
 * dst[i] = src[i] - alpha
 */
template <typename Dtype>
void sub_scalar(Dtype alpha, const Array<Dtype>& src, Array<Dtype>* dst) {
  for (int i = 0; i < src.total_; i++) {
    dst->d_[i] = src.d_[i] - alpha;
  }
}

/**
 * dst[i] = src[i] - alpha
 */
template <typename Dtype>
void sub_scalar(int n, Dtype alpha, const Dtype* src, Dtype* dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = src[i] - alpha;
  }
}

}  // namespace cnn
