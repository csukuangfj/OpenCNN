#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/array_math.hpp"
#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class L2LossLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        proto.set_type(L2_LOSS);
        layer_ = Layer<Dtype>::create(proto);
    }
protected:
    std::shared_ptr<Layer<Dtype>> layer_;

    Array<Dtype> bottom1_;  //!< the prediction
    Array<Dtype> bottom2_;  //!< the ground truth

    Array<Dtype> bottom1_gradient_; //!< gradient for the predication
    Array<Dtype> top_;      //!< the loss, which is a scalar, i.e., with shape (1,1,1,1)
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(L2LossLayerTest, MyTypes);

TYPED_TEST(L2LossLayerTest, reshape)
{
    // the train phase
    this->layer_->proto().set_phase(TRAIN);
    this->bottom1_.init(2, 3, 4, 5);
    this->bottom2_.init_like(this->bottom1_);

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});

    EXPECT_TRUE(this->bottom1_gradient_.has_same_shape(this->bottom1_));
    EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));

    // now the test phase, no gradient memory should be allocated
    this->layer_->proto().set_phase(TEST);
    this->bottom1_gradient_.init(0, 0, 0, 0);
    this->top_.init(0, 0, 0, 0);
    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});
    EXPECT_TRUE(this->bottom1_gradient_.has_same_shape({0, 0, 0, 0}));
    EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));
}

TYPED_TEST(L2LossLayerTest, fprop)
{
    this->bottom1_.init(2, 3, 4, 5);
    this->bottom2_.init(2, 3, 4, 5);

    set_to<TypeParam>(&this->bottom1_, 2);
    set_to<TypeParam>(&this->bottom2_, 2);

    int n = uniform(1, this->bottom1_.total_);
    for (int i = 0; i < n; i++)
    {
        this->bottom2_[i] = 1;
    }

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});

    this->layer_->fprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->top_});
    EXPECT_NEAR(this->top_[0], TypeParam(n)/this->bottom1_.total_, 1e-4);
}

TYPED_TEST(L2LossLayerTest, bprop)
{
    this->bottom1_.init(2, 5, 3, 4);
    this->bottom2_.init(2, 5, 3, 4);

    set_to<TypeParam>(&this->bottom1_, 2);
    set_to<TypeParam>(&this->bottom2_, 2);

    int n = uniform(1, this->bottom1_.total_-1);
    for (int i = 0; i < n; i++)
    {
        this->bottom2_[i] = 1;
    }

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});

    this->layer_->fprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->top_});

    this->layer_->bprop(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});
    for (int i = 0; i < this->bottom1_gradient_.total_; i++)
    {
        if (i < n)
        {
            EXPECT_EQ(this->bottom1_gradient_[i], 1);
        }
        else
        {
            EXPECT_EQ(this->bottom1_gradient_[i], 0);
        }
    }
}


}  // namespace cnn

