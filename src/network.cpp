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

#include <string>
#include <vector>

#include "cnn/io.hpp"
#include "cnn/network.hpp"

namespace cnn {
template <typename Dtype>
Network<Dtype>::Network(const NetworkProto& _proto) {
  init(_proto);
}

template <typename Dtype>
Network<Dtype>::Network(const std::string& filename,
                        bool is_binary /*= false*/) {
  init(filename, is_binary);
}

template <typename Dtype>
void Network<Dtype>::init(const std::string& filename,
                          bool is_binary /*= false*/) {
  NetworkProto network_proto;
  if (is_binary) {
    read_proto_bin(filename, &network_proto);
  } else  // NOLINT
  {
    read_proto_txt(filename, &network_proto);
  }

  // LOG(INFO) << layer_proto.DebugString();
  init(network_proto);
}

template <typename Dtype>
void Network<Dtype>::init(const NetworkProto& _proto) {
  proto_ = _proto;
  LOG(INFO) << "\n" << proto_.DebugString();
  // a network MUST have at least two
  // layers: an input layer and an output layer
  CHECK_GE(proto_.layer_proto_size(), 2);

  const auto& input_layer = proto_.layer_proto(0);
  CHECK_EQ(input_layer.type(), INPUT)
      << "The 0th layer has to be of type INPUT!";

  CHECK_EQ(input_layer.bottom_size(), 0)
      << "Input layer should have no bottom!";

  LOG(INFO) << "process layer: " << input_layer.name();
  // allocate space for the input
  for (int i = 0; i < input_layer.top_size(); i++) {
    auto d = std::make_shared<Array<Dtype>>();
    add_data(input_layer.top(i), d);

    // add it for convenience; it is never referenced
    auto g = std::make_shared<Array<Dtype>>();
    add_gradient(input_layer.top(i), g);
  }

  layers_.push_back(Layer<Dtype>::create(input_layer));

  // create other layers
  for (int i = 1; i < proto_.layer_proto_size(); i++) {
    // check its bottom has been created!
    const auto& layer_proto = proto_.layer_proto(i);
    LOG(INFO) << "process layer " << layer_proto.name();
    for (int j = 0; j < layer_proto.bottom_size(); j++) {
      const auto& name = layer_proto.bottom(j);
      CHECK_EQ(data_.count(name), 1)
          << "bottom with name " << name << " does not exist!";
    }

    // then creates its top
    for (int j = 0; j < layer_proto.top_size(); j++) {
      auto d = std::make_shared<Array<Dtype>>();
      add_data(layer_proto.top(j), d);

      auto g = std::make_shared<Array<Dtype>>();
      add_gradient(layer_proto.top(j), g);
    }

    layers_.push_back(Layer<Dtype>::create(layer_proto));
  }
}

template <typename Dtype>
void Network<Dtype>::copy_trained_network(const std::string& filename,
                                          bool is_binary /*= false*/) {
  NetworkProto network_proto;
  if (is_binary) {
    read_proto_bin(filename, &network_proto);
  } else  // NOLINT
  {
    read_proto_txt(filename, &network_proto);
  }

  for (int i = 0; i < network_proto.layer_proto_size(); i++) {
    const auto& p = network_proto.layer_proto(i);
    if (!p.param_size()) {
      continue;
    }

    for (auto& _layer : layers_) {
      if (_layer->proto().name() == p.name()) {
        _layer->copy_trained_layer(p);
      }
    }
  }
}

template <typename Dtype>
void Network<Dtype>::save_network(const std::string& filename,
                                  bool is_binary /*= false*/) {
  int num_layers = layers_.size();
  NetworkProto _proto;
  for (int i = 0; i < num_layers; i++) {
    auto& layer = *layers_[i];
    auto* target = _proto.add_layer_proto();
    target->CopyFrom(layer.proto());
    target->clear_param();

    auto param = layer.param();
    for (int j = 0; j < param.size(); j++) {
      param[j]->to_proto(target->add_param());
    }
  }

  if (is_binary) {
    write_proto_bin(filename, _proto);
  } else  // NOLINT
  {
    write_proto_txt(filename, _proto);
  }
}

template <typename Dtype>
void Network<Dtype>::reshape() {
  layers_[0]->reshape({}, {}, get_data_top_mutable(0), {});
  for (int i = 1; i < layers_.size(); i++) {
    LOG(INFO) << "layer " << layers_[i]->proto().name() << " reshape()";
    layers_[i]->reshape(get_data_bottom(i), get_gradient_bottom_mutable(i),
                        get_data_top_mutable(i), get_gradient_top_mutable(i));
    for (const auto& b : get_data_bottom(i)) {
      LOG(INFO) << "  " << b->shape_info();
    }
  }
}

template <typename Dtype>
void Network<Dtype>::fprop() {
  if (data_callback_) {
    data_callback_(get_data_top_mutable(0));
  } else  // NOLINT
  {
    layers_[0]->fprop({}, get_data_top_mutable(0));
  }
  for (int i = 1; i < layers_.size(); i++) {
    layers_[i]->fprop(get_data_bottom(i), get_data_top_mutable(i));
  }
}

template <typename Dtype>
void Network<Dtype>::bprop() {
  for (int i = 0; i < layers_.size(); i++) {
    layers_[i]->clear_gradient();
  }

  for (auto& g : gradient_) {
    set_to<Dtype>(g.second.get(), 0);
  }

  layers_[layers_.size() - 1]->bprop(
      get_data_bottom(layers_.size() - 1),
      get_gradient_bottom_mutable(layers_.size() - 1),
      get_data_top(layers_.size() - 1), {});
  std::ostringstream ss;
  ss << "gradient for the last layer:\n";
  const auto* g = get_gradient_bottom(layers_.size() - 1)[0];
  for (int i = 0; i < g->total_; i++) {
    ss << g->d_[i] << " ";
  }
  ss << "\n";
  // LOG(INFO) << ss.str();

  for (int i = layers_.size() - 2; i >= 1; i--) {
    layers_[i]->bprop(get_data_bottom(i), get_gradient_bottom_mutable(i),
                      get_data_top(i), get_gradient_top(i));
  }
}

template <typename Dtype>
void Network<Dtype>::set_phase(Phase phase) {
  if (phase_ == phase) {
    return;
  }

  for (auto& _layer : layers_) {
    _layer->proto().set_phase(phase);
  }

  phase_ = phase;
}

template <typename Dtype>
void Network<Dtype>::perform_predication() {
  auto saved_phase = phase_;
  set_phase(TEST);

  // we assume that the user has already setup the input data
  // via get_data_top(0)
  for (int i = 1; i < layers_.size(); i++) {
    layers_[i]->fprop(get_data_bottom(i), get_data_top_mutable(i));
  }
  set_phase(saved_phase);
}

template <typename Dtype>
void Network<Dtype>::add_data(const std::string& name,
                              std::shared_ptr<Array<Dtype>> arr) {
  CHECK_EQ(data_.count(name), 0) << "duplicate name " << name;
  data_[name] = arr;
  LOG(INFO) << "add: " << name;
}

template <typename Dtype>
void Network<Dtype>::add_gradient(const std::string& name,
                                  std::shared_ptr<Array<Dtype>> arr) {
  CHECK_EQ(gradient_.count(name), 0) << "duplicate name " << name;
  gradient_[name] = arr;
  LOG(INFO) << "add gradient: " << name;
}

template <typename Dtype>
std::vector<const Array<Dtype>*> Network<Dtype>::get_data_bottom(int i) const {
  std::vector<const Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.bottom_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.bottom(i);
    CHECK_EQ(data_.count(name), 1)
        << "data with name " << name << " does not exist!";
    res.push_back(data_.at(name).get());
  }
  return res;
}

template <typename Dtype>
std::vector<Array<Dtype>*> Network<Dtype>::get_gradient_bottom_mutable(int i) {
  std::vector<Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.bottom_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.bottom(i);
    CHECK_EQ(gradient_.count(name), 1)
        << "gradient with name " << name << " does not exist!";
    res.push_back(gradient_[name].get());
  }
  return res;
}

template <typename Dtype>
std::vector<const Array<Dtype>*> Network<Dtype>::get_gradient_bottom(
    int i) const {
  std::vector<const Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.bottom_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.bottom(i);
    CHECK_EQ(gradient_.count(name), 1)
        << "gradient with name " << name << " does not exist!";
    res.push_back(gradient_.at(name).get());
  }
  return res;
}

template <typename Dtype>
std::vector<const Array<Dtype>*> Network<Dtype>::get_data_top(int i) const {
  std::vector<const Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.top_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.top(i);
    CHECK_EQ(data_.count(name), 1)
        << "data with name " << name << " does not exist!";
    res.push_back(data_.at(name).get());
  }
  return res;
}

template <typename Dtype>
std::vector<Array<Dtype>*> Network<Dtype>::get_data_top_mutable(int i) {
  std::vector<Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.top_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.top(i);
    CHECK_EQ(data_.count(name), 1)
        << "data with name " << name << " does not exist!";
    res.push_back(data_[name].get());
  }
  return res;
}

template <typename Dtype>
std::vector<const Array<Dtype>*> Network<Dtype>::get_gradient_top(int i) const {
  std::vector<const Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.top_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.top(i);
    CHECK_EQ(gradient_.count(name), 1)
        << "gradient with name " << name << " does not exist!";
    res.push_back(gradient_.at(name).get());
  }
  return res;
}

template <typename Dtype>
std::vector<Array<Dtype>*> Network<Dtype>::get_gradient_top_mutable(int i) {
  std::vector<Array<Dtype>*> res;
  const auto& layer_proto = layers_[i]->proto();
  int n = layer_proto.top_size();
  for (int i = 0; i < n; i++) {
    const auto& name = layer_proto.top(i);
    CHECK_EQ(gradient_.count(name), 1)
        << "gradient with name " << name << " does not exist!";
    res.push_back(gradient_[name].get());
  }
  return res;
}

}  // namespace cnn
