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

#include <memory>
#include <string>
#include <vector>

#include "proto/cnn.pb.h"

#include "cnn/network.hpp"

namespace cnn {

template <typename Dtype>
class Optimizer {
 public:
  explicit Optimizer(const OptimizerProto& _proto);
  explicit Optimizer(const std::string& filename);
  void init(const OptimizerProto& _proto);

  void start_training();

  void register_data_callback(void (*f)(const std::vector<Array<Dtype>*>&)) {
    network_->register_data_callback(f);
  }

 private:
  void update_parameters(int current_iter);

 private:
  void print_parameters();

 private:
  OptimizerProto proto_;

  std::shared_ptr<Network<Dtype>> network_;
};

}  // namespace cnn

#include "../../src/optimizer.cpp"
