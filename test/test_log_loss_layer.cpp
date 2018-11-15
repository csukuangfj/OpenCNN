#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class LogLossLayerTest : public ::testing::Test
{
    void SetUp() override
    {
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

TYPED_TEST(LogLossLayerTest, reshape_train_phase)
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

TYPED_TEST(LogLossLayerTest, reshape_test_phase)
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

TYPED_TEST(LogLossLayerTest, fprop)
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

    this->layer_->fprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->top_});

    LOG(INFO) << "loss is: " << this->top_[0];
    TypeParam expected;
    expected = -(std::log(0.25) + std::log(0.875)
                    + std::log(1e-20) + std::log(0.25));
    expected /= 4;

    EXPECT_NEAR(this->top_[0], expected, 1e-5);
}

TYPED_TEST(LogLossLayerTest, bprop)
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

    this->layer_->fprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->top_});
    this->layer_->bprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});
    const auto& bg = this->bottom1_gradient_;

    TypeParam scale = - TypeParam(1.)/4;

    // batch 0
    EXPECT_EQ(bg[0], scale / b1[0]);
    EXPECT_EQ(bg[1], 0);
    EXPECT_EQ(bg[2], 0);
    EXPECT_EQ(bg[3], scale / b1[3]);

    // batch 1
    EXPECT_EQ(bg[4], 0);
    EXPECT_EQ(bg[5], 0);

    // NOTE (fangjun): the input probability should not be 0
    // otherwise, its derivative is very very huge!
    TypeParam tmp = scale/TypeParam(1e-20);
    TypeParam tmp2 = bg[6]/tmp;
    EXPECT_NEAR(tmp2, 1, 1e-5);
    EXPECT_EQ(bg[7], scale / b1[7]);
}


}  // cnn

