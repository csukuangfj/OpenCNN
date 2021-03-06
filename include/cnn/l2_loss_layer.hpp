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
 * For regression only!
 * It computes
 *
 * \frac{1}{n} \sum_{i=0}^{n-1} (\mathrm{bottom}[0][i] -
 * \mathrm{bottom}[1][i])^2
 *
 * bottom[0] is the predication and bottom[1] is the ground truth.
 * Both of them have shape (N, C, H, W).
 */
template <typename Dtype>
class L2LossLayer : public Layer<Dtype> {
 public:
  explicit L2LossLayer(const LayerProto&);

  /**
   * @param bottom bottom[0] is the predication and bottom[1] is the ground
   * truth; they must have the same shape
   * @param bottom_gradient the gradient for bottom[0].
   *               It is allocated only in the train phase
   * @param top   the average squared loss between bottom[0] and bottom[1]
   * @param top_gradient  it is not used and we always assume that the top
   * gradient is 1
   */
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
  Dtype loss_;
};

}  // namespace cnn

#include "../../src/l2_loss_layer.cpp"
