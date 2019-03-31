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
 * top[0]->d_[i] = max(0, bottom[0]->d_[i]) + alpha_ * min(0, bottom[0]->d_[i]);
 *
 * Refer to
 *  - "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
 *  - https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
 *
 * The purpose of leaky ReLU is to propagate gradient even if
 * the input is negative.
 *
 * The gradient for the negative input is a non-negative constant.
 *
 */
template <typename Dtype>
class LeakyReLULayer : public Layer<Dtype> {
 public:
  explicit LeakyReLULayer(const LayerProto&);

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
  Dtype alpha_;  //!<  greater than or equal to 0
                 //!< It cannot be negative, otherwise a negative input
                 //!< results in a positive output!
};

}  // namespace cnn

#include "../../src/leaky_relu_layer.cpp"
