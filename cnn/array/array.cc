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

#include <sstream>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "cnn/array/array.h"

namespace cnn {

template <typename Dtype>
Array<Dtype>::Array() : n_(0), c_(0), h_(0), w_(0), total_(0), d_(nullptr) {}

template <typename Dtype>
Array<Dtype>::~Array() {
  if (d_) delete[] d_;
}

template <typename Dtype>
Array<Dtype>::Array(Array<Dtype>&& arr) {
  n_ = arr.n_;
  c_ = arr.c_;
  h_ = arr.h_;
  w_ = arr.w_;
  total_ = arr.total_;
  d_ = arr.d_;

  arr.n_ = 0;
  arr.c_ = 0;
  arr.h_ = 0;
  arr.w_ = 0;
  arr.total_ = 0;
  arr.d_ = nullptr;
}

template <typename Dtype>
Array<Dtype>& Array<Dtype>::operator=(Array<Dtype>&& arr) {
  if (this == &arr) {
    return *this;
  }

  if (d_) {
    delete[] d_;
  }

  n_ = arr.n_;
  c_ = arr.c_;
  h_ = arr.h_;
  w_ = arr.w_;
  total_ = arr.total_;
  d_ = arr.d_;

  arr.n_ = 0;
  arr.c_ = 0;
  arr.h_ = 0;
  arr.w_ = 0;
  arr.total_ = 0;
  arr.d_ = nullptr;

  return *this;
}

template <typename Dtype>
template <typename U>
void Array<Dtype>::init_like(const Array<U>& arr) {
  if (this->has_same_shape(arr)) return;
  init(arr.n_, arr.c_, arr.h_, arr.w_);
}

template <typename Dtype>
void Array<Dtype>::init(int n, int c, int h, int w) {
  CHECK_GE(n, 0);
  CHECK_GE(c, 0);
  CHECK_GE(h, 0);
  CHECK_GE(w, 0);

  int total = n * c * h * w;
  if (total == 0) {
    if (d_) {
      delete[] d_;
      d_ = nullptr;
    }
    n_ = c_ = h_ = w_ = 0;
    total_ = 0;
    return;
  }

  if (total != total_) {
    if (d_) delete[] d_;

    d_ = new Dtype[n * c * h * w];
  }

  memset(d_, 0, total * sizeof(Dtype));
  n_ = n;
  c_ = c;
  h_ = h;
  w_ = w;
  total_ = total;
}

template <typename Dtype>
template <typename U>
bool Array<Dtype>::has_same_shape(const Array<U>& arr) const {
  bool res;
  res = (total_ == arr.total_) && (n_ == arr.n_) && (c_ == arr.c_) &&
        (h_ == arr.h_) && (w_ == arr.w_);
  return res;
}

template <typename Dtype>
bool Array<Dtype>::has_same_shape(const std::vector<int>& vec) const {
  bool res;
  res = (vec.size() == 4) && (vec[0] == n_) && (vec[1] == c_) &&
        (vec[2] == h_) && (vec[3] == w_);
  return res;
}

template <typename Dtype>
const Dtype& Array<Dtype>::at(int n, int c, int h, int w) const {
  CHECK_GE(n, 0);
  CHECK_LT(n, n_);
  CHECK_GE(c, 0);
  CHECK_LT(c, c_);
  CHECK_GE(h, 0);
  CHECK_LT(h, h_);
  CHECK_GE(w, 0);
  CHECK_LT(w, w_);
  int i = ((n * c_ + c) * h_ + h) * w_ + w;
  return d_[i];
}

template <typename Dtype>
Dtype& Array<Dtype>::at(int n, int c, int h, int w) {
  CHECK_GE(n, 0);
  CHECK_LT(n, n_);
  CHECK_GE(c, 0);
  CHECK_LT(c, c_);
  CHECK_GE(h, 0);
  CHECK_LT(h, h_);
  CHECK_GE(w, 0);
  CHECK_LT(w, w_);
  int i = ((n * c_ + c) * h_ + h) * w_ + w;
  return d_[i];
}

template <typename Dtype>
const Dtype& Array<Dtype>::operator()(int n, int c, int h, int w) const {
  int i = ((n * c_ + c) * h_ + h) * w_ + w;
  return d_[i];
}

template <typename Dtype>
Dtype& Array<Dtype>::operator()(int n, int c, int h, int w) {
  int i = ((n * c_ + c) * h_ + h) * w_ + w;
  return d_[i];
}

template <typename Dtype>
const Dtype& Array<Dtype>::operator[](int i) const {
  return d_[i];
}

template <typename Dtype>
Dtype& Array<Dtype>::operator[](int i) {
  return d_[i];
}

template <typename Dtype>
std::string Array<Dtype>::shape_info() const {
  std::ostringstream ss;
  ss << n_ << ", " << c_ << ", " << h_ << ", " << w_ << "\n";
  return ss.str();
}

template <typename Dtype>
void Array<Dtype>::from_proto(const ArrayProto& proto) {
  init(proto.n(), proto.c(), proto.h(), proto.w());
  for (int i = 0; i < total_; i++) {
    d_[i] = static_cast<Dtype>(proto.d(i));
  }
}

template <typename Dtype>
void Array<Dtype>::to_proto(ArrayProto* proto) const {
  proto->set_n(n_);
  proto->set_c(c_);
  proto->set_h(h_);
  proto->set_w(w_);

  proto->clear_d();

  for (int i = 0; i < total_; i++) {
    proto->add_d(d_[i]);
  }
}

template class Array<float>;
template class Array<double>;
template class Array<bool>;

}  // namespace cnn
