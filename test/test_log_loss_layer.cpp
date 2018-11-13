#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/log_loss_layer.hpp"

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

}  // cnn

