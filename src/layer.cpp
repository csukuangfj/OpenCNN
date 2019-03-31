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

#include "cnn/batch_normalization_layer.hpp"
#include "cnn/convolution_layer.hpp"
#include "cnn/drop_out_layer.hpp"
#include "cnn/full_connected_layer.hpp"
#include "cnn/input_layer.hpp"
#include "cnn/jet.hpp"
#include "cnn/l2_loss_layer.hpp"
#include "cnn/layer.hpp"
#include "cnn/leaky_relu_layer.hpp"
#include "cnn/log_loss_layer.hpp"
#include "cnn/max_pooling_layer.hpp"
#include "cnn/relu_layer.hpp"
#include "cnn/softmax_layer.hpp"
#include "cnn/softmax_with_log_loss_layer.hpp"

namespace cnn {
template <typename Dtype>
Layer<Dtype>::Layer(const LayerProto& _proto) : param_(), proto_(_proto) {
  if (proto_.param_size()) {
    param_.clear();
    for (int i = 0; i < proto_.param_size(); i++) {
      auto arr = std::make_shared<Array<Dtype>>();
      arr->from_proto(proto_.param(i));
      param_.push_back(arr);
    }
  }
}

template <typename Dtype>
std::shared_ptr<Layer<Dtype>> Layer<Dtype>::create(const LayerProto& _proto) {
#define CREATE_LAYER(type_name, class_name)   \
  case type_name:                             \
    res.reset(new class_name<Dtype>(_proto)); \
    break

  std::shared_ptr<Layer<Dtype>> res;
  switch (_proto.type()) {
    CREATE_LAYER(INPUT, InputLayer);
    CREATE_LAYER(FULL_CONNECTED, FullConnectedLayer);
    CREATE_LAYER(L2_LOSS, L2LossLayer);
    CREATE_LAYER(SOFTMAX, SoftmaxLayer);
    CREATE_LAYER(LOG_LOSS, LogLossLayer);
    CREATE_LAYER(SOFTMAX_WITH_LOG_LOSS, SoftmaxWithLogLossLayer);
    CREATE_LAYER(CONVOLUTION, ConvolutionLayer);
    CREATE_LAYER(RELU, ReLULayer);
    CREATE_LAYER(MAX_POOLING, MaxPoolingLayer);
    CREATE_LAYER(DROP_OUT, DropoutLayer);
    CREATE_LAYER(BATCH_NORMALIZATION, BatchNormalizationLayer);
    CREATE_LAYER(LEAKY_RELU, LeakyReLULayer);

    default:
      LOG(FATAL) << "Unknown layer type: " << LayerType_Name(_proto.type());
      break;
  }

#undef CREATE_LAYER

  CHECK_NOTNULL(res.get());
  return res;
}

template <typename Dtype>
void Layer<Dtype>::copy_trained_layer(const LayerProto& p) {
  CHECK_EQ(proto_.name(), p.name());
  CHECK_EQ(proto_.type(), p.type());
  CHECK_EQ(proto_.top_size(), p.top_size());
  CHECK_EQ(proto_.bottom_size(), p.bottom_size());

  if (param_.size()) {
    CHECK_EQ(param_.size(), p.param_size());
    param_.clear();
  }

  for (int i = 0; i < p.param_size(); i++) {
    auto arr = std::make_shared<Array<Dtype>>();
    arr->from_proto(p.param(i));
    param_.push_back(arr);
  }
}
template <typename Dtype>
void Layer<Dtype>::update_parameters(int /*current_iter*/,
                                     double current_learning_rate) {
  if (gradient_.empty()) {
    // this layer has no parameters, skip it.
    // For example, the softmax layer has no parameters
    return;
  }
  if (history_gradient_.empty()) {
    history_gradient_.resize(gradient_.size());
    for (int i = 0; i < gradient_.size(); i++) {
      history_gradient_[i].reset(new Array<Dtype>);
      history_gradient_[i]->init_like(*gradient_[i]);
    }
  }

  // TODO(fangjun): move the following options to proto
  static const Dtype decay = 0.0000;
  static const Dtype momentum = 0.0;
  // TODO(fangjun):
  // gradient = gradient + decay*param;
  // history = momentum*history + (1-momentum)*gradient
  // param = param - learning_rate*history;
  //
  // it usually writes
  // history = momentum*history + gradient
  // param = param - learning_rate*history
  //
  // momentum is typically 0.9 or 0.99
  // refer to lecture 7 of CS231N at stanford
  for (int i = 0; i < gradient_.size(); i++) {
    // apply weight decay
    ax_plus_by<Dtype>(param_[i]->total_, decay, &param_[i]->d_[0], 1,
                      &gradient_[i]->d_[0]);

    // update history
    ax_plus_by<Dtype>(param_[i]->total_, current_learning_rate,
                      &gradient_[i]->d_[0], momentum,
                      &history_gradient_[i]->d_[0]);

    // update parameters
    ax_plus_by<Dtype>(param_[i]->total_, -1, &history_gradient_[i]->d_[0], 1,
                      &param_[i]->d_[0]);
  }
}

}  // namespace cnn
