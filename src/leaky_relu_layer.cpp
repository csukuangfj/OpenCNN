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

#include <vector>

#include "cnn/jet.hpp"
#include "cnn/relu_layer.hpp"

namespace cnn {

template <typename Dtype>
LeakyReLULayer<Dtype>::LeakyReLULayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto) {
  alpha_ = _proto.leaky_relu_proto().alpha();
  CHECK_GE(alpha_, 0);
}

template <typename Dtype>
void LeakyReLULayer<Dtype>::reshape(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<Array<Dtype>*>& top,
    const std::vector<Array<Dtype>*>& top_gradient) {
  CHECK_EQ(bottom.size(), 1) << "leaky relu accepts only 1 input";
  CHECK_EQ(top.size(), 1) << "leaky relu generates only 1 output";

  top[0]->init_like(*bottom[0]);

  if (this->proto_.phase() == TRAIN) {
    CHECK_EQ(bottom_gradient.size(), 1);

    if (!bottom_gradient[0]->has_same_shape(*bottom[0])) {
      bottom_gradient[0]->init_like(*bottom[0]);
    }

    CHECK_EQ(top_gradient.size(), 1);
    top_gradient[0]->init_like(*top[0]);
  }
}

template <typename Dtype>
void LeakyReLULayer<Dtype>::fprop(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& top) {
  const auto& b = *bottom[0];
  auto& t = *top[0];
  for (int i = 0; i < b.total_; i++) {
    t[i] = ((b[i] >= Dtype(0)) + (b[i] < Dtype(0)) * alpha_) * b[i];
  }
}

template <typename Dtype>
void LeakyReLULayer<Dtype>::bprop(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<const Array<Dtype>*>& /*top*/,
    const std::vector<const Array<Dtype>*>& top_gradient) {
  const auto& b = *bottom[0];
  auto& bg = *bottom_gradient[0];

  const auto& tg = *top_gradient[0];

  for (int i = 0; i < b.total_; i++) {
    bg[i] = tg[i] * ((b[i] >= Dtype(0)) + (b[i] < Dtype(0)) * alpha_);
  }
}

}  // namespace cnn
