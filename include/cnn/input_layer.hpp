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

template <typename Dtype>
class InputLayer : public Layer<Dtype> {
 public:
  explicit InputLayer(const LayerProto&);

  void reshape(const std::vector<const Array<Dtype>*>& bottom,
               const std::vector<Array<Dtype>*>& bottom_gradient,
               const std::vector<Array<Dtype>*>& top,
               const std::vector<Array<Dtype>*>& top_gradient) override;

  void fprop(const std::vector<const Array<Dtype>*>& input,
             const std::vector<Array<Dtype>*>& output) override;

  void bprop(const std::vector<const Array<Dtype>*>&,
             const std::vector<Array<Dtype>*>&,
             const std::vector<const Array<Dtype>*>&,
             const std::vector<const Array<Dtype>*>&) override {}

 private:
  int n_;
  int c_;
  int h_;
  int w_;
};

}  // namespace cnn

#include "../../src/input_layer.cpp"
