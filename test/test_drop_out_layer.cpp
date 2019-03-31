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
#include <gtest/gtest.h>

#define private public

#include "cnn/jet.hpp"
#include "cnn/layer.hpp"

namespace cnn {

template <typename Dtype>
class DropoutLayerTest : public ::testing::Test {
  void SetUp() override {
    LayerProto proto;
    proto.set_type(DROP_OUT);
    auto* p = proto.mutable_dropout_proto();
    p->set_keep_prob(0.8);
    layer_ = Layer<Dtype>::create(proto);
  }

 protected:
  std::shared_ptr<Layer<Dtype>> layer_;

  Array<Dtype> bottom_;
  Array<Dtype> bottom_gradient_;
  Array<Dtype> top_;
  Array<Dtype> top_gradient_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(DropoutLayerTest, MyTypes);

TYPED_TEST(DropoutLayerTest, reshape_train_phase) {
  static constexpr int N = 2;
  static constexpr int C = 3;
  static constexpr int H = 4;
  static constexpr int W = 5;

  this->layer_->proto().set_phase(TRAIN);

  this->bottom_.init(N, C, H, W);
  this->layer_->reshape({&this->bottom_}, {&this->bottom_gradient_},
                        {&this->top_}, {&this->top_gradient_});

  EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));

  EXPECT_TRUE(this->top_.has_same_shape(this->bottom_));
  EXPECT_TRUE(this->top_gradient_.has_same_shape(this->bottom_));

  EXPECT_TRUE(this->layer_->param().empty());
  EXPECT_TRUE(this->layer_->gradient().empty());

  auto* dropout_layer =
      dynamic_cast<DropoutLayer<TypeParam>*>(this->layer_.get());
  EXPECT_TRUE(dropout_layer->mask_.has_same_shape(this->top_));
}

TYPED_TEST(DropoutLayerTest, reshape_test_phase) {
  static constexpr int N = 2;
  static constexpr int C = 3;
  static constexpr int H = 4;
  static constexpr int W = 5;

  this->layer_->proto().set_phase(TEST);

  this->bottom_.init(N, C, H, W);
  this->layer_->reshape({&this->bottom_}, {&this->bottom_gradient_},
                        {&this->top_}, {&this->top_gradient_});

  EXPECT_TRUE(this->bottom_gradient_.has_same_shape({0, 0, 0, 0}));

  EXPECT_TRUE(this->top_.has_same_shape(this->bottom_));
  EXPECT_TRUE(this->top_gradient_.has_same_shape({0, 0, 0, 0}));

  EXPECT_TRUE(this->layer_->param().empty());
  EXPECT_TRUE(this->layer_->gradient().empty());

  auto* dropout_layer =
      dynamic_cast<DropoutLayer<TypeParam>*>(this->layer_.get());
  EXPECT_TRUE(dropout_layer->mask_.has_same_shape({0, 0, 0, 0}));
}

// we do not need to test the fprop since it is implicitly
// verified in the bprop below.
TYPED_TEST(DropoutLayerTest, bprop_with_jet) {
  return;
  static constexpr int N = 2;
  static constexpr int C = 3;
  static constexpr int H = 4;
  static constexpr int W = 5;
  static constexpr int DIM = 1;

  using Type = Jet<TypeParam, DIM>;

  LayerProto proto;
  proto.set_phase(TRAIN);
  proto.set_type(DROP_OUT);
  proto.mutable_dropout_proto()->set_keep_prob(0.5);
  auto layer = Layer<Type>::create(proto);

  Array<Type> bottom;
  Array<Type> bottom_gradient;
  Array<Type> top;
  Array<Type> top_gradient;

  bottom.init(N, C, H, W);

  uniform<Type>(&bottom, -100, 100);
  for (int i = 0; i < bottom.total_; i++) {
    bottom.d_[i].v_[0] = 1;
  }

  layer->reshape({&bottom}, {&bottom_gradient}, {&top}, {&top_gradient});

  uniform<Type>(&top_gradient, -100, 100);
  layer->fprop({&bottom}, {&top});
  layer->bprop({&bottom}, {&bottom_gradient}, {&top}, {&top_gradient});

  for (int i = 0; i < bottom_gradient.total_; i++) {
    TypeParam expected = top[i].v_[0] * top_gradient[i].a_;
    if (expected == 0) {
      EXPECT_EQ(bottom_gradient[i], 0);
    } else {
      EXPECT_NEAR(bottom_gradient[i].a_ / expected, 1, 1e-4);
    }
  }
}

}  // namespace cnn
