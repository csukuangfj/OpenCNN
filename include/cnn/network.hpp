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

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "proto/cnn.pb.h"

#include "cnn/array.hpp"
#include "cnn/layer.hpp"

namespace cnn {
template <typename Dtype>
class Network {
 public:
  explicit Network(const NetworkProto&);
  explicit Network(const std::string& filename, bool is_binary = false);
  void init(const std::string& filename, bool is_binary = false);
  void init(const NetworkProto&);

  void copy_trained_network(const std::string& filename,
                            bool is_binary = false);

  const NetworkProto& proto() const { return proto_; }
  NetworkProto& proto() { return proto_; }

  void reshape();
  /** forward propagation */
  void fprop();

  /** backward propagation */
  void bprop();

  void set_phase(Phase phase);
  Phase get_phase() const { return phase_; }

  Dtype get_loss() const { return get_data_top(layers_.size() - 1)[0]->d_[0]; }
  /**
   * we assume that the output of the last blob saves the predication
   */

  void perform_predication();
  std::vector<Dtype> get_predications() const {
    std::vector<Dtype> res;
    auto top = get_data_top(layers_.size() - 1)[0];
    for (int i = 0; i < top->total_; i++) {
      res.push_back(top->d_[i]);
    }
    return res;
  }

  int get_batch_size() const { return layers_[0]->proto().input_proto().n(); }

  void save_network(const std::string& filename, bool is_binary = false);

  std::shared_ptr<Layer<Dtype>> layer(int i) const { return layers_[i]; }
  const std::vector<std::shared_ptr<Layer<Dtype>>>& layers() const {
    return layers_;
  }
  std::vector<std::shared_ptr<Layer<Dtype>>>& layers() { return layers_; }

  std::vector<const Array<Dtype>*> get_data_bottom(int i) const;
  std::vector<const Array<Dtype>*> get_data_top(int i) const;
  std::vector<Array<Dtype>*> get_data_top_mutable(int i);

  std::vector<Array<Dtype>*> get_gradient_bottom_mutable(int i);
  std::vector<const Array<Dtype>*> get_gradient_bottom(int i) const;
  std::vector<const Array<Dtype>*> get_gradient_top(int i) const;
  std::vector<Array<Dtype>*> get_gradient_top_mutable(int i);

  void register_data_callback(void (*f)(const std::vector<Array<Dtype>*>&)) {
    data_callback_ = f;
  }

 private:
  // add data to the map
  void add_data(const std::string& name, std::shared_ptr<Array<Dtype>> arr);

  void add_gradient(const std::string& name, std::shared_ptr<Array<Dtype>> arr);

 private:
  NetworkProto proto_;

  /** it saves the input and output of all layers in the network*/
  std::map<std::string, std::shared_ptr<Array<Dtype>>> data_;
  std::map<std::string, std::shared_ptr<Array<Dtype>>> gradient_;

  std::vector<std::shared_ptr<Layer<Dtype>>> layers_;

  std::function<void(const std::vector<Array<Dtype>*>& top)> data_callback_;

  Phase phase_;
};

}  // namespace cnn

#include "../../src/network.cpp"
