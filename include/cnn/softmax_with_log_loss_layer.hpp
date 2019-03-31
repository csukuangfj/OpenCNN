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
#include "cnn/softmax_layer.hpp"

namespace cnn {
/**
 * Softmax + log loss.
 *
 * Motivation:
 *  -(1/y) * (y - y*y) does not equal to -(1 - y)
 *  when y is close to 0 from a programmer's perspective
 *  of view!!! I learn this by hard.
 *  It causes gradient to be 0 when y is tiny,
 *  i.e., the vanishing gradient problem.
 *
 * For classification only!
 *
 * It accepts two inputs: bottom[0] and bottom[1].
 *
 * bottom[0] is the prediction with shape (N, C, H, W).
 * The meaning of the shape can be interpreted as follows:
 * (1) N is the batch size. (2) H and W can be considered
 * as the size of an image. (3) C represents the number of channels
 * of the image; in the case of multi-class classification, C
 * is the number of classes we have and the meaning of each pixel
 * in the channel k indicates the probability to the k-th category
 * this pixel belongs.
 *
 * The user has to ensure that the pixel values are in
 * the range of [0, 1] and it is normalized, i.e., sum to 1.
 *
 * This layer lays usually on top of the full connected layer.
 *
 * bottom[1] is the ground truth and has the shape (N, 1, H, W).
 * values of every element in bottom[1] have to be an integer
 * in the range [0, C-1].
 */
template <typename Dtype>
class SoftmaxWithLogLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLogLossLayer(const LayerProto&);

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
  std::shared_ptr<Layer<Dtype>> softmax_layer_;
  Array<Dtype> softmax_top_;
  Array<Dtype> softmax_top_gradient_;
  Array<Dtype> softmax_bottom_gradient_;
};

}  // namespace cnn

#include "../../src/softmax_with_log_loss_layer.cpp"
