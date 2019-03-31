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

#include "cnn/array_math.hpp"
#include "cnn/jet.hpp"
#include "cnn/layer.hpp"

namespace cnn {

template <typename Dtype>
class LeakyReLULayerTest : public ::testing::Test {
  void SetUp() override {
    LayerProto proto;
    proto.set_type(LEAKY_RELU);
    proto.mutable_leaky_relu_proto()->set_alpha(alpha_);

    layer_ = Layer<Dtype>::create(proto);
  }

 protected:
  std::shared_ptr<Layer<Dtype>> layer_;

  Array<Dtype> bottom_;
  Array<Dtype> bottom_gradient_;
  Array<Dtype> top_;
  Array<Dtype> top_gradient_;

  Dtype alpha_ = 0.3;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LeakyReLULayerTest, MyTypes);

TYPED_TEST(LeakyReLULayerTest, reshape_train_phase) {
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

  auto* layer = dynamic_cast<LeakyReLULayer<TypeParam>*>(this->layer_.get());
  CHECK_NOTNULL(layer);
  CHECK_EQ(layer->alpha_, this->alpha_);
}

TYPED_TEST(LeakyReLULayerTest, reshape_test_phase) {
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

  auto* layer = dynamic_cast<LeakyReLULayer<TypeParam>*>(this->layer_.get());
  CHECK_NOTNULL(layer);
  CHECK_EQ(layer->alpha_, this->alpha_);
}

TYPED_TEST(LeakyReLULayerTest, fprop) {
  static constexpr int N = 2;
  static constexpr int C = 3;
  static constexpr int H = 4;
  static constexpr int W = 5;

  this->layer_->proto().set_phase(TEST);

  this->bottom_.init(N, C, H, W);
  uniform<TypeParam>(&this->bottom_, -100, 100);
  this->layer_->reshape({&this->bottom_}, {}, {&this->top_}, {});

  this->layer_->fprop({&this->bottom_}, {&this->top_});

  TypeParam expected;

  const auto& b = this->bottom_;
  const auto& t = this->top_;

  for (int i = 0; i < b.total_; i++) {
    if (b[i] >= 0) {
      expected = b[i];
    } else {
      expected = b[i] * this->alpha_;
    }
    EXPECT_EQ(t[i], expected);
  }
}

TYPED_TEST(LeakyReLULayerTest, bprop_with_jet) {
  static constexpr int N = 2;
  static constexpr int C = 3;
  static constexpr int H = 4;
  static constexpr int W = 5;
  static constexpr int DIM = 1;

  using Type = Jet<TypeParam, DIM>;

  LayerProto proto;
  proto.set_phase(TRAIN);
  proto.set_type(LEAKY_RELU);
  proto.mutable_leaky_relu_proto()->set_alpha(0.01);
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
    EXPECT_EQ(bottom_gradient[i].a_, expected);
  }
}

}  // namespace cnn
