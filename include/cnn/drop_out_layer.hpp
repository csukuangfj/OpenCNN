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
 * It has one bottom and one top.
 *
 * Both bottom[0] and top[0] have shape (N, C, H, W).
 *
 * During the train phase,
 *
 * top[0]->d_[i] = bottom[0]->d_[i] * mask_[i] / keep_prob_;
 *
 * where mask[i] is either 1 or 0. Its probability to be 1 is keep_prob_;
 *
 * During the test phase,
 *
 * top[0]->d_[i] = bottom[0]->d_[i];
 *
 * We use inverted dropout here.
 *
 * Refer to
 * http://cs231n.github.io/neural-networks-2/#reg
 *
 * Dropout is short for DROP OUTput.
 * A similar word is DropConnect.
 */
template <typename Dtype>
class DropoutLayer : public Layer<Dtype> {
 public:
  explicit DropoutLayer(const LayerProto&);

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
  Dtype keep_prob_;
  Array<bool> mask_;
};

}  // namespace cnn

#include "../../src/drop_out_layer.cpp"
