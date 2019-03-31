#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/layer.hpp"

namespace cnn {

template <typename Dtype>
class LogLossLayerTest : public ::testing::Test {
  void SetUp() override {
    LayerProto proto;
    proto.set_type(LOG_LOSS);
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
TYPED_TEST_CASE(LogLossLayerTest, MyTypes);

TYPED_TEST(LogLossLayerTest, reshape_train_phase) {
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

TYPED_TEST(LogLossLayerTest, reshape_test_phase) {
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

TYPED_TEST(LogLossLayerTest, fprop) {
  this->layer_->proto().set_phase(TRAIN);

  this->bottom1_.init(2, 2, 1, 2);
  this->bottom2_.init(2, 1, 1, 2);

  this->layer_->reshape({&this->bottom1_, &this->bottom2_},
                        {&this->bottom1_gradient_}, {&this->top_}, {});
  auto& b1 = this->bottom1_;
  auto& b2 = this->bottom2_;

  /*
   * predication:
   * batch 0
   * 0.25  0.125
   * 0.75  0.875
   *
   * batch 1
   * 1  0.75
   * 0  0.25
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
  b1[0] = TypeParam(0.25);
  b1[1] = TypeParam(0.125);
  b1[2] = TypeParam(0.75);
  b1[3] = TypeParam(0.875);

  b1[4] = TypeParam(1);
  b1[5] = TypeParam(0.75);
  b1[6] = TypeParam(0);
  b1[7] = TypeParam(0.25);

  b2[0] = 0;
  b2[1] = 1;
  b2[2] = 1;
  b2[3] = 1;

  this->layer_->fprop({&this->bottom1_, &this->bottom2_}, {&this->top_});

  // LOG(INFO) << "loss is: " << this->top_[0];
  TypeParam expected;
  expected =
      -(std::log(0.25) + std::log(0.875) + std::log(1e-20) + std::log(0.25));
  expected /= 4;

  EXPECT_NEAR(this->top_[0], expected, 1e-5);
}

TYPED_TEST(LogLossLayerTest, bprop_with_jet) {
  static constexpr int N = 6;
  static constexpr int C = 8;
  static constexpr int H = 2;
  static constexpr int W = 3;
  static constexpr int DIM = N * C * H * W;

  using Type = Jet<TypeParam, DIM>;

  LayerProto proto;
  proto.set_phase(TRAIN);
  proto.set_type(LOG_LOSS);
  auto layer = Layer<Type>::create(proto);

  Array<Type> bottom1;
  Array<Type> bottom2;
  Array<Type> bottom1_gradient;
  Array<Type> top;

  bottom1.init(N, C, H, W);
  bottom2.init(N, 1, H, W);

  uniform<Type>(&bottom1, 20, 50);
  uniform<Type>(&bottom2, 0, C - 1);  // labels are in the range [0, C-1]

  // normalize across channels
  for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++) {
        Type t = 0;
        for (int c = 0; c < C; c++) {
          t += bottom1(n, c, h, w);
        }

        for (int c = 0; c < C; c++) {
          bottom1(n, c, h, w) /= t;
        }
      }

  for (int i = 0; i < bottom1.total_; i++) {
    bottom1[i].v_[i] = 1;
  }

  layer->reshape({&bottom1, &bottom2}, {&bottom1_gradient}, {&top}, {});
  layer->fprop({&bottom1, &bottom2}, {&top});
  layer->bprop({&bottom1, &bottom2}, {&bottom1_gradient}, {&top}, {});
  for (int i = 0; i < bottom1_gradient.total_; i++) {
    EXPECT_NEAR(bottom1_gradient[i].a_, top[0].v_[i], 1e-6);
  }
}

}  // namespace cnn
