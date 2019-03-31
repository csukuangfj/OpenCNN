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
#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn {
/**
 * It has one input bottom[0] with shape (N, C, H, W)
 * and one output top[0] with shape (N, M, 1, 1),
 * where M is the number of output provided in the prototxt.
 *
 * param[0] has the shape (1, 1, M, K) where K equals to C*H*W
 * and param[1] has shape (1, 1, 1, M).
 *
 * param[0] contains weight parameters for inner product
 * and param[1] contains corresponding biases.
 */
template <typename Dtype>
class FullConnectedLayer : public Layer<Dtype> {
 public:
  explicit FullConnectedLayer(const LayerProto&);

  void reshape(const std::vector<const Array<Dtype>*>& bottom,
               const std::vector<Array<Dtype>*>& bottom_gradient,
               const std::vector<Array<Dtype>*>& top,
               const std::vector<Array<Dtype>*>& top_gradient) override;

  void fprop(const std::vector<const Array<Dtype>*>& bottom,
             const std::vector<Array<Dtype>*>& top) override;

  void bprop(const std::vector<const Array<Dtype>*>& bottom,
             const std::vector<Array<Dtype>*>& bottom_gradient,
             const std::vector<const Array<Dtype>*>& top,
             const std::vector<const Array<Dtype>*>& top_gradient) override;

 private:
  int num_output_;
};

}  // namespace cnn

#include "../../src/full_connected_layer.cpp"
