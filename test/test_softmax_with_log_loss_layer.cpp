#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/log_loss_layer.hpp"

namespace cnn
{

template<typename Dtype>
class SoftmaxWithLogLossLayerTest : public ::testing::Test
{
    void SetUp() override
    {
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

TYPED_TEST(SoftmaxWithLogLossLayerTest, reshape_train_phase)
{
    // the train phase
    this->layer_->proto().set_phase(TRAIN);

    this->bottom1_.init(2, 3, 4, 5);    // 3 means we have 3 categories
    this->bottom2_.init(2, 1, 4, 5);

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});

    EXPECT_TRUE(this->bottom1_gradient_.has_same_shape(this->bottom1_));
    EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(SoftmaxWithLogLossLayerTest, reshape_test_phase)
{
    // the test phase
    this->layer_->proto().set_phase(TEST);

    this->bottom1_.init(2, 6, 5, 8);    // 6 means we have 6 categories
    this->bottom2_.init(2, 1, 5, 8);

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});

    EXPECT_TRUE(this->bottom1_gradient_.has_same_shape({0, 0, 0, 0}));
    EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(SoftmaxWithLogLossLayerTest, fprop)
{
    this->layer_->proto().set_phase(TRAIN);

    this->bottom1_.init(2, 2, 1, 2);
    this->bottom2_.init(2, 1, 1, 2);

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});
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

    this->layer_->fprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->top_});

    TypeParam expected = 0;
    expected -= 3 + 2 + 8 + 10;
    expected += std::log(std::exp(3) + std::exp(6));
    expected += std::log(std::exp(2) + std::exp(5));
    expected += std::log(std::exp(-2) + std::exp(8));
    expected += std::log(std::exp(1) + std::exp(10));

    expected /= 4;

    LOG(INFO) << "loss is: " << this->top_[0];
    LOG(INFO) << "expected is : " << this->top_[0];

    EXPECT_NEAR(this->top_[0], expected, 1e-5);
}

TYPED_TEST(SoftmaxWithLogLossLayerTest, bprop)
{
    this->layer_->proto().set_phase(TRAIN);

    this->bottom1_.init(2, 2, 1, 2);
    this->bottom2_.init(2, 1, 1, 2);

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});
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
 * after the softmax operation,
 * 0.0474259 0.952574
 * 0.952574 0.0474259
 *
 * 4.53979e-05 0.000123395
 * 0.999955 0.999877
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

    this->layer_->fprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->top_});


    this->layer_->bprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});
    const auto& bg = this->bottom1_gradient_;

    TypeParam scale = TypeParam(1.)/4;
/*
 *
 *
 * after the softmax operation,
 * 0.0474259 0.952574
 * 0.952574 0.0474259
 *
 * 4.53979e-05 0.000123395
 * 0.999955 0.999877
 */

    // batch 0
    EXPECT_NEAR(bg[0], scale * (0.0474259 - 1), 1e-5);
    EXPECT_NEAR(bg[1], scale*0.952574, 1e-5);
    EXPECT_NEAR(bg[2], scale*0.952574, 1e-5);
    EXPECT_NEAR(bg[3], scale*(0.0474259 - 1), 1e-5);

    EXPECT_NEAR(bg[4], scale * 4.53979e-5, 1e-5);
    EXPECT_NEAR(bg[5], scale*0.000123395, 1e-5);
    EXPECT_NEAR(bg[6], scale*(0.999955 - 1), 1e-5);
    EXPECT_NEAR(bg[7], scale*(0.999877 - 1), 1e-5);
}

}  // cnn


