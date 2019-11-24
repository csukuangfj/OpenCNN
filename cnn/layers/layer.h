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
#ifndef CNN_LAYERS_LAYER_H_
#define CNN_LAYERS_LAYER_H_

#include <memory>
#include <vector>

#include "cnn/array/array.h"
#include "cnn/array/array_math.h"
#include "cnn/proto/cnn.pb.h"

// TODO(fangjun): replace shared_ptr with unique_ptr

namespace cnn {
/**
 *
 * Every layer MUST implement the following functions
 *  * reshape
 *  * fprop
 *  * bprop
 */
template <typename Dtype>
class Layer {
 public:
  explicit Layer(const LayerProto&);
  static std::shared_ptr<Layer<Dtype>> create(const LayerProto&);

  const LayerProto& proto() const { return proto_; }
  LayerProto& proto() { return proto_; }

  std::vector<Array<Dtype>*> mutable_param() {
    std::vector<Array<Dtype>*> res;
    for (int i = 0; i < param_.size(); i++) {
      res.push_back(param_[i].get());
    }
    return res;
  }

  std::vector<const Array<Dtype>*> param() const {
    std::vector<const Array<Dtype>*> res;
    for (int i = 0; i < param_.size(); i++) {
      res.push_back(param_[i].get());
    }
    return res;
  }

  std::vector<Array<Dtype>*> mutable_gradient() {
    std::vector<Array<Dtype>*> res;
    for (int i = 0; i < gradient_.size(); i++) {
      res.push_back(gradient_[i].get());
    }
    return res;
  }

  std::vector<const Array<Dtype>*> gradient() const {
    std::vector<const Array<Dtype>*> res;
    for (int i = 0; i < gradient_.size(); i++) {
      res.push_back(gradient_[i].get());
    }
    return res;
  }

  void clear_gradient() {
    for (auto& g : gradient_) {
      if (g) {
        set_to<Dtype>(g.get(), 0);
      }
    }
  }

  void copy_trained_layer(const LayerProto& p);

  void update_parameters(int current_iter, double current_learning_rate);

  /**
   * At layer construction, we have no idea of the shape of its inputs,
   * so this function MUST be called after constructing the whole network.
   */
  virtual void reshape(const std::vector<const Array<Dtype>*>& bottom,
                       const std::vector<Array<Dtype>*>& bottom_gradient,
                       const std::vector<Array<Dtype>*>& top,
                       const std::vector<Array<Dtype>*>& top_gradient) = 0;

  /**
   * forward propagation
   */
  virtual void fprop(const std::vector<const Array<Dtype>*>& bottom,
                     const std::vector<Array<Dtype>*>& top) = 0;

  /**
   * backward propagation
   */
  virtual void bprop(const std::vector<const Array<Dtype>*>& bottom,
                     const std::vector<Array<Dtype>*>& bottom_gradient,
                     const std::vector<const Array<Dtype>*>& top,
                     const std::vector<const Array<Dtype>*>& top_gradient) = 0;

 protected:
  std::vector<std::shared_ptr<Array<Dtype>>> param_;
  std::vector<std::shared_ptr<Array<Dtype>>> gradient_;
  std::vector<std::shared_ptr<Array<Dtype>>> history_gradient_;

  LayerProto proto_;

 private:
  Layer(const Layer<Dtype>&) = delete;
  Layer& operator=(const Layer<Dtype>&) = delete;
};

}  // namespace cnn

#endif  // CNN_LAYERS_LAYER_H_
