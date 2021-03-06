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
#include <glog/logging.h>

#include <utility>
#include <vector>

#include "cnn/input_layer.hpp"

namespace cnn {
template <typename Dtype>
InputLayer<Dtype>::InputLayer(const LayerProto& _proto) : Layer<Dtype>(_proto) {
  const auto param = _proto.input_proto();
  n_ = param.n();
  c_ = param.c();
  h_ = param.h();
  w_ = param.w();
}

template <typename Dtype>
void InputLayer<Dtype>::reshape(
    const std::vector<const Array<Dtype>*>& /*bottom*/,
    const std::vector<Array<Dtype>*>& /*bottom_gradient*/,
    const std::vector<Array<Dtype>*>& top,
    const std::vector<Array<Dtype>*>& /*top_gradient*/) {
  CHECK((top.size() == 1) || (top.size() == 2));

  top[0]->init(n_, c_, h_, w_);
  if (top.size() == 2) {
    // resize the label
    top[1]->init(n_, 1, 1, 1);
  }
}

template <typename Dtype>
void InputLayer<Dtype>::fprop(
    const std::vector<const Array<Dtype>*>& /*bottom*/,
    const std::vector<Array<Dtype>*>& top) {
  // y = 5 + 10*x
  static std::vector<std::pair<std::vector<Dtype>, Dtype>> data{
      {{11}, 115}, {{-14}, -135}, {{15}, 155}, {{6}, 65},   {{-18}, -175},
      {{-8}, -75}, {{9}, 95},     {{-4}, -35}, {{18}, 185}, {{-1}, -5},
  };
  // TODO(fangjun) load data from here

  int n = top[0]->n_;
  int stride = top[0]->total_ / n;
  if (stride != data[0].first.size()) return;

  static int k = 0;
  CHECK_LE(n, data.size())
      << "the batch size cannot be larger than the dataset size";

  CHECK_EQ(stride, data[0].first.size());

  for (int i = 0; i < n; i++) {
    if (k >= data.size()) {
      k = 0;
    }

    for (int j = 0; j < stride; j++) {
      top[0]->d_[i * stride + j] = (data[k].first)[j];
    }

    if (top.size() == 2) {
      top[1]->d_[i] = data[k].second;
    }

    k++;
  }
}

}  // namespace cnn
