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

#include "cnn/layer.hpp"

namespace cnn {

template <typename Dtype>
class SoftmaxWithLogLossLayerTest : public ::testing::Test {
  void SetUp() override {
    LayerProto proto;
    proto.set_type(SOFTMAX_WITH_LOG_LOSS);
    layer_ = Layer<Dtype>::create(proto);
  }

 protected:
  std::shared_ptr<Layer<Dtype>> layer_;

  Array<Dtype> bottom1_;
  Array<Dtype> bottom2_;
  Array<Dtype> bottom1_gradient_;
  Array<Dtype> top_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(SoftmaxWithLogLossLayerTest, MyTypes);

TYPED_TEST(SoftmaxWithLogLossLayerTest, reshape_train_phase) {
  // the train phase
  this->layer_->proto().set_phase(TRAIN);

  this->bottom1_.init(2, 3, 4, 5);  // 3 means we have 3 categories
  this->bottom2_.init(2, 1, 4, 5);

  this->layer_->reshape({&this->bottom1_, &this->bottom2_},
                        {&this->bottom1_gradient_}, {&this->top_}, {});

  EXPECT_TRUE(this->bottom1_gradient_.has_same_shape(this->bottom1_));
  EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));

  EXPECT_TRUE(this->layer_->param().empty());
  EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(SoftmaxWithLogLossLayerTest, reshape_test_phase) {
  // the test phase
  this->layer_->proto().set_phase(TEST);

  this->bottom1_.init(2, 6, 5, 8);  // 6 means we have 6 categories
  this->bottom2_.init(2, 1, 5, 8);

  this->layer_->reshape({&this->bottom1_, &this->bottom2_},
                        {&this->bottom1_gradient_}, {&this->top_}, {});

  EXPECT_TRUE(this->bottom1_gradient_.has_same_shape({0, 0, 0, 0}));
  EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));

  EXPECT_TRUE(this->layer_->param().empty());
  EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(SoftmaxWithLogLossLayerTest, fprop) {
  this->layer_->proto().set_phase(TRAIN);

  this->bottom1_.init(2, 2, 1, 2);
  this->bottom2_.init(2, 1, 1, 2);

  this->layer_->reshape({&this->bottom1_, &this->bottom2_},
                        {&this->bottom1_gradient_}, {&this->top_}, {});
  auto& b1 = this->bottom1_;
  auto& b2 = this->bottom2_;

  /*
   * input
   * batch 0
   * 3 5
   * 6 2
   *
   * batch 1
   * -2  1
   * 8  10
   *
   * ground truth
   *
   * batch 0
   *
   * 0  1
   *
   * batch 1
   * 1  1
   */
  b1[0] = TypeParam(3);
  b1[1] = TypeParam(5);
  b1[2] = TypeParam(6);
  b1[3] = TypeParam(2);

  b1[4] = TypeParam(-2);
  b1[5] = TypeParam(1);
  b1[6] = TypeParam(8);
  b1[7] = TypeParam(10);

  b2[0] = 0;
  b2[1] = 1;
  b2[2] = 1;
  b2[3] = 1;

  this->layer_->fprop({&this->bottom1_, &this->bottom2_}, {&this->top_});

  TypeParam expected = 0;
  expected -= 3 + 2 + 8 + 10;
  expected += std::log(std::exp(3) + std::exp(6));
  expected += std::log(std::exp(2) + std::exp(5));
  expected += std::log(std::exp(-2) + std::exp(8));
  expected += std::log(std::exp(1) + std::exp(10));

  expected /= 4;

  // LOG(INFO) << "loss is: " << this->top_[0];
  // LOG(INFO) << "expected is : " << this->top_[0];

  EXPECT_NEAR(this->top_[0], expected, 1e-5);
}

TYPED_TEST(SoftmaxWithLogLossLayerTest, bprop_with_jet) {
  static constexpr int N = 2;
  static constexpr int C = 3;
  static constexpr int H = 4;
  static constexpr int W = 5;
  static constexpr int DIM = N * C * H * W;

  using Type = Jet<TypeParam, DIM>;

  LayerProto proto;
  proto.set_phase(TRAIN);
  proto.set_type(SOFTMAX_WITH_LOG_LOSS);
  auto layer = Layer<Type>::create(proto);

  Array<Type> bottom1;
  Array<Type> bottom2;
  Array<Type> bottom1_gradient;
  Array<Type> top;

  bottom1.init(N, C, H, W);
  bottom2.init(N, 1, H, W);

  // the test might be broken for float types if the values in
  // bottom1 vary significantly, i.e., from 20 to 50
  // that is, uniform<Type>(&bottom1, 20, 50);
  // fails for float, but fine for double
  uniform<Type>(&bottom1, 2, 8);
  uniform<Type>(&bottom2, 0, C - 1);  // labels are in the range [0, C-1]

  for (int i = 0; i < bottom1.total_; i++) {
    bottom1[i].v_[i] = 1;
  }

  layer->reshape({&bottom1, &bottom2}, {&bottom1_gradient}, {&top}, {});

  layer->fprop({&bottom1, &bottom2}, {&top});

  layer->bprop({&bottom1, &bottom2}, {&bottom1_gradient}, {&top}, {});
  for (int i = 0; i < bottom1_gradient.total_; i++) {
    EXPECT_NEAR(bottom1_gradient[i].a_, top[0].v_[i], 1e-5);
  }
}

}  // namespace cnn
