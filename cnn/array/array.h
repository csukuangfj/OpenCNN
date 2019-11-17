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
#ifndef CNN_ARRAY_ARRAY_H_
#define CNN_ARRAY_ARRAY_H_

#include <string>
#include <vector>

#include "cnn/proto/cnn.pb.h"

namespace cnn {

template <typename Dtype>
class Array {
 public:
  Array();
  ~Array();
  Array(const Array<Dtype>&) = delete;
  Array& operator=(const Array<Dtype>&) = delete;

  Array(Array<Dtype>&&);
  Array& operator=(Array<Dtype>&&);

  void init(int n, int c, int h, int w);

  template <typename U>
  void init_like(const Array<U>& arr);

  template <typename U>
  bool has_same_shape(const Array<U>& arr) const;

  bool has_same_shape(const std::vector<int>& vec) const;
  std::vector<int> shape_vec() const { return {n_, c_, h_, w_}; }

  /**
   * Return the element at n*c_*h_*w_ + c*h_*w_ + h*w_ + w,
   * i.e., (n*c_ + c)*h_*w_ + h*w_ + w,
   * i.e., ((n*c_ + c)*h_ + h)*w_ + w
   */
  const Dtype& at(int n, int c, int h, int w) const;
  Dtype& at(int n, int c, int h, int w);

  // no range check
  const Dtype& operator()(int n, int c, int h, int w) const;
  Dtype& operator()(int n, int c, int h, int w);

  const Dtype& operator[](int i) const;
  Dtype& operator[](int i);

  std::string shape_info() const;

  int n_;      //!< number of batches
  int c_;      //!< number of channels
  int h_;      //!< image height, i.e., number of rows
  int w_;      //!< image width, number of columns
  int total_;  //!< n_*c_*h_*w_, number of elements

  Dtype* d_;  //!< pointer to the data

 public:
  void from_proto(const ArrayProto& proto);
  void to_proto(ArrayProto* proto) const;
};

extern template class Array<float>;
extern template class Array<double>;

}  // namespace cnn

#endif  // CNN_ARRAY_ARRAY_H_
