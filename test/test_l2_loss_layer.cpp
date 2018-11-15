#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/array_math.hpp"
#include "cnn/layer.hpp"
#include "cnn/jet.hpp"

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

TYPED_TEST(L2LossLayerTest, reshape_train_phase)
{
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

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(L2LossLayerTest, reshape_test_phase)
{
    this->layer_->proto().set_phase(TEST);
    this->bottom1_.init(2, 3, 4, 5);
    this->bottom2_.init_like(this->bottom1_);

    this->layer_->reshape(
            {&this->bottom1_, &this->bottom2_},
            {&this->bottom1_gradient_},
            {&this->top_},
            {});

    EXPECT_TRUE(this->top_.has_same_shape({1, 1, 1, 1}));
    EXPECT_TRUE(this->bottom1_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());
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

TYPED_TEST(L2LossLayerTest, bprop_with_jet)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;
    static constexpr int DIM = N*C*H*W;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(L2_LOSS);
    auto layer = Layer<Jet<TypeParam, DIM>>::create(proto);

    using Type = Jet<TypeParam, DIM>;

    Array<Type> bottom1;
    Array<Type> bottom2;
    Array<Type> bottom1_gradient;
    Array<Type> top;

    bottom1.init(N, C, H, W);
    bottom2.init(N, C, H, W);

    uniform<Type>(&bottom1, -100, 100);
    uniform<Type>(&bottom2, -100, 100);
    for (int i = 0; i < bottom1.total_; i++)
    {
        bottom1.d_[i].v_[i] = 1;
    }

    layer->reshape(
            {&bottom1, &bottom2},
            {&bottom1_gradient},
            {&top},
            {});

    layer->fprop({&bottom1, &bottom2}, {&top});
    layer->bprop(
            {&bottom1, &bottom2},
            {&bottom1_gradient},
            {&top},
            {});

    for (int i = 0; i < bottom1.total_; i++)
    {
        EXPECT_NEAR(bottom1_gradient[i].a_,
                  top[0].v_[i], 1e-5);
    }
}

}  // namespace cnn

