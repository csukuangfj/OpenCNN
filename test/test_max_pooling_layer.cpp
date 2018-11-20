#include <glog/logging.h>
#include <gtest/gtest.h>

#define private public
#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class MaxPoolingLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        auto* p = proto.mutable_max_pooling_proto();
        p->set_win_size(2);
        p->set_stride(2);
        proto.set_type(MAX_POOLING);
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
TYPED_TEST_CASE(MaxPoolingLayerTest, MyTypes);

TYPED_TEST(MaxPoolingLayerTest, reshape_train_phase)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 10;
    static constexpr int W = 9;

    this->layer_->proto().set_phase(TRAIN);

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));

    EXPECT_TRUE(this->top_.has_same_shape({N, C, 5, 4}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(this->top_));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());

    const auto* layer = dynamic_cast<MaxPoolingLayer<TypeParam>*>(this->layer_.get());
    EXPECT_TRUE(layer->max_index_pair_.has_same_shape(this->top_));
}

TYPED_TEST(MaxPoolingLayerTest, reshape_test_phase)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 9;
    static constexpr int W = 10;

    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_TRUE(this->top_.has_same_shape({N, C, 4, 5}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());

    const auto* layer = dynamic_cast<MaxPoolingLayer<TypeParam>*>(this->layer_.get());
    EXPECT_TRUE(layer->max_index_pair_.has_same_shape(this->top_));
}

TYPED_TEST(MaxPoolingLayerTest, fprop)
{
    static constexpr int N = 1;
    static constexpr int C = 1;
    static constexpr int H = 4;
    static constexpr int W = 4;

    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(N, C, H, W);
    TypeParam b[C*H*W] = {
        // channel 0
        0, -1, 2, 9,
        3, 8, -3, 7,
        -3, 2, 5, -2,
        7, 3, -8, 6
    };
    for (int i = 0; i < C*H*W; i++)
    {
        this->bottom_[i] = b[i];
    }

    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    this->layer_->fprop(
            {&this->bottom_},
            {&this->top_});

    TypeParam expected;

    const auto& t = this->top_;

    EXPECT_EQ(t[0], 8);
    EXPECT_EQ(t[1], 9);
    EXPECT_EQ(t[2], 7);
    EXPECT_EQ(t[3], 6);
}

TYPED_TEST(MaxPoolingLayerTest, bprop_with_jet)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 10;
    static constexpr int W = 10;
    static constexpr int DIM = N*C*H*W;

    using Type = Jet<TypeParam, DIM>;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(MAX_POOLING);
    auto* p = proto.mutable_max_pooling_proto();
    p->set_win_size(3);
    p->set_stride(2);
    auto layer = Layer<Type>::create(proto);

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);

    uniform<Type>(&bottom, -100, 100);
    for (int i = 0; i < DIM; i++)
    {
        bottom.d_[i].v_[i] = 1;
    }

    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    uniform<Type>(&top_gradient, -100, 100);
    layer->fprop({&bottom}, {&top});
    layer->bprop(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    Type s = TypeParam(0);
    for (int i = 0; i < top.total_; i++)
    {
        s += top[i]*top_gradient[i].a_;
    }

    for (int i = 0; i < DIM; i++)
    {
        TypeParam expected = s.v_[i];
        EXPECT_EQ(bottom_gradient[i].a_, expected);
    }
}

}  // namespace cnn

