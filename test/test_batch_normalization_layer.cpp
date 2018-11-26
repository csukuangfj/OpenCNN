#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/layer.hpp"
#include "cnn/jet.hpp"

namespace cnn
{

template<typename Dtype>
class BatchNormalizationLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        proto.set_type(BATCH_NORMALIZATION);
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
TYPED_TEST_CASE(BatchNormalizationLayerTest, MyTypes);

TYPED_TEST(BatchNormalizationLayerTest, reshape_train_phase)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;

    this->layer_->proto().set_phase(TRAIN);

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));

    EXPECT_TRUE(this->top_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(this->bottom_));

    EXPECT_EQ(this->layer_->param().size(), 4);
    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(this->layer_->param()[i]->has_same_shape({1, 3, 1, 1}));
    }

    EXPECT_EQ(this->layer_->gradient().size(), 2);
    for (int i = 0; i < 2; i++)
    {
        EXPECT_TRUE(this->layer_->gradient()[i]->has_same_shape({1, 3, 1, 1}));
    }
}

TYPED_TEST(BatchNormalizationLayerTest, reshape_test_phase)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;

    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_TRUE(this->top_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_EQ(this->layer_->param().size(), 4);
    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(this->layer_->param()[i]->has_same_shape({1, 3, 1, 1}));
    }

    EXPECT_EQ(this->layer_->gradient().size(), 0);
}

TYPED_TEST(BatchNormalizationLayerTest, fprop_train)
{
    this->layer_->proto().set_phase(TRAIN);

    static constexpr int N = 2;
    static constexpr int C = 2;
    static constexpr int H = 1;
    static constexpr int W = 2;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});
    // scale
    this->layer_->param()[0]->d_[0] = 1;
    this->layer_->param()[0]->d_[1] = 2;

    // bias
    this->layer_->param()[1]->d_[0] = 3;
    this->layer_->param()[1]->d_[1] = 4;

    TypeParam a[N*C*H*W] = {
        // batch 0, channel 0
        10, 2,
        // batch 0, channel 1
        -1, 9,

        // batch 1, channel 0
        5, -20,
        // batch 1, channel 1
        6, 12,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = a[i];
    }

    this->layer_->fprop({&this->bottom_}, {&this->top_});
/*
import numpy as np
a = np.array([10, 2, -1, 9, 5, -20, 6, 12]).reshape(2, 2, 1, 2)
mu = np.mean(a, axis=(0, 2, 3), keepdims=True)
std = np.std(a, axis=(0, 2, 3), keepdims=True)
c = (a - mu)/std

gamma = np.array([1, 2]).reshape(1, 2, 1, 1)
beta = np.array([3, 4]).reshape(1, 2, 1, 1)
print(c*gamma + beta)
//------
[[[[3.93677693 3.23964061]]

  [[0.88914492 5.03695169]]]


 [[[3.50106673 1.32251573]]

  [[3.79260966 6.28129373]]]]
 */
    const auto&t = this->top_;
    EXPECT_NEAR(t[0], 3.93677693, 1e-6);
    EXPECT_NEAR(t[1], 3.23964061, 1e-6);
    EXPECT_NEAR(t[2], 0.88914492, 1e-6);
    EXPECT_NEAR(t[3], 5.03695169, 1e-6);
    EXPECT_NEAR(t[4], 3.50106673, 1e-6);
    EXPECT_NEAR(t[5], 1.32251573, 1e-6);
    EXPECT_NEAR(t[6], 3.79260966, 1e-6);
    EXPECT_NEAR(t[7], 6.28129373, 1e-6);
}

TYPED_TEST(BatchNormalizationLayerTest, fprop_test)
{
    this->layer_->proto().set_phase(TRAIN);

    static constexpr int N = 2;
    static constexpr int C = 2;
    static constexpr int H = 1;
    static constexpr int W = 2;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});
    // scale
    this->layer_->param()[0]->d_[0] = 1;
    this->layer_->param()[0]->d_[1] = 2;

    // bias
    this->layer_->param()[1]->d_[0] = 3;
    this->layer_->param()[1]->d_[1] = 4;

    TypeParam a[N*C*H*W] = {
        // batch 0, channel 0
        10, 2,
        // batch 0, channel 1
        -1, 9,

        // batch 1, channel 0
        5, -20,
        // batch 1, channel 1
        6, 12,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = a[i];
    }

    this->layer_->fprop({&this->bottom_}, {&this->top_});

    this->layer_->proto().set_phase(TEST);
    TypeParam b[N*C*H*W] = {
        // batch 0, channel 0
        1, 0,
        // batch 0, channel 1
        20, 3,

        // batch 1, channel 0
        8, 6,
        // batch 1, channel 1
        5, 20,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = b[i];
    }

    this->layer_->fprop({&this->bottom_}, {&this->top_});

/*
a = np.array([10, 2, -1, 9, 5, -20, 6, 12]).reshape(2, 2, 1, 2)
mu = np.mean(a, axis=(0, 2, 3), keepdims=True)
std = np.std(a, axis=(0, 2, 3), keepdims=True)

b = np.array([1, 0, 20, 3, 8, 6, 5, 20]).reshape(2, 2, 1, 2)
c = (b - mu)/std

gamma = np.array([1, 2]).reshape(1, 2, 1, 1)
beta = np.array([3, 4]).reshape(1, 2, 1, 1)
print(c*gamma + beta)
//-----
[[[[3.15249857 3.06535653]]

  [[9.59953915 2.54826763]]]


 [[[3.76249285 3.58820877]]

  [[3.37782898 9.59953915]]]]
 */

    const auto&t = this->top_;
    EXPECT_NEAR(t[0], 3.15249857, 1e-6);
    EXPECT_NEAR(t[1], 3.06535653, 1e-6);
    EXPECT_NEAR(t[2], 9.59953915, 1e-5);
    EXPECT_NEAR(t[3], 2.54826763, 1e-6);
    EXPECT_NEAR(t[4], 3.76249285, 1e-6);
    EXPECT_NEAR(t[5], 3.58820877, 1e-6);
    EXPECT_NEAR(t[6], 3.37782898, 1e-6);
    EXPECT_NEAR(t[7], 9.59953915, 1e-5);
}

TYPED_TEST(BatchNormalizationLayerTest, fprop_train_test)
{
    this->layer_->proto().set_phase(TRAIN);

    static constexpr int N = 2;
    static constexpr int C = 2;
    static constexpr int H = 1;
    static constexpr int W = 2;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});
    // scale
    this->layer_->param()[0]->d_[0] = 1;
    this->layer_->param()[0]->d_[1] = 2;

    // bias
    this->layer_->param()[1]->d_[0] = 3;
    this->layer_->param()[1]->d_[1] = 4;

    TypeParam a[N*C*H*W] = {
        // batch 0, channel 0
        10, 2,
        // batch 0, channel 1
        -1, 9,

        // batch 1, channel 0
        5, -20,
        // batch 1, channel 1
        6, 12,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = a[i];
    }
    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam b[N*C*H*W] = {
        // batch 0, channel 0
        1, 0,
        // batch 0, channel 1
        20, 3,

        // batch 1, channel 0
        8, 6,
        // batch 1, channel 1
        5, 20,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = b[i];
    }

    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam c[N*C*H*W] = {
        // batch 0, channel 0
        -10, 3,
        // batch 0, channel 1
        -5, 12,

        // batch 1, channel 0
        22, 60,
        // batch 1, channel 1
        -2, 10,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = c[i];
    }

    this->layer_->fprop({&this->bottom_}, {&this->top_});

    this->layer_->proto().set_phase(TEST);

    TypeParam d[N*C*H*W] = {
        // batch 0, channel 0
        1, 7,
        // batch 0, channel 1
        -9, 18,

        // batch 1, channel 0
        -2, -6,
        // batch 1, channel 1
        25, 50,
    };

    for (int i = 0; i < this->bottom_.total_; i++)
    {
        this->bottom_[i] = d[i];
    }

    this->layer_->fprop({&this->bottom_}, {&this->top_});

/*
import numpy as np
a = np.array([10, 2, -1, 9, 5, -20, 6, 12]).reshape(2, 2, 1, 2)
mu = np.mean(a, axis=(0, 2, 3), keepdims=True)
std = np.std(a, axis=(0, 2, 3), keepdims=True)

b = np.array([1, 0, 20, 3, 8, 6, 5, 20]).reshape(2, 2, 1, 2)
mu2 = np.mean(b, axis=(0, 2, 3), keepdims=True)
std2 = np.std(b, axis=(0, 2, 3), keepdims=True)

c = np.array([-10, 3, -5, 12, 22, 60, -2, 10]).reshape(2, 2, 1, 2)
mu3 = np.mean(c, axis=(0, 2, 3), keepdims=True)
std3 = np.std(c, axis=(0, 2, 3), keepdims=True)

momentum = 0.99

mu = mu*momentum + mu2*(1-momentum)
mu = mu*momentum + mu3*(1-momentum)

std = std*momentum + std2*(1-momentum)
std = std*momentum + std3*(1-momentum)

d = np.array([1, 7, -9, 18, -2, -6, 25, 50]).reshape(2, 2, 1, 2)

f = (d - mu)/std

gamma = np.array([1, 2]).reshape(1, 2, 1, 1)
beta = np.array([3, 4]).reshape(1, 2, 1, 1)
print(f*gamma + beta)

//----
[[[[ 3.13084041  3.65058119]]

  [[-2.36481881  8.70304113]]]


 [[[ 2.87097002  2.52447616]]

  [[11.5724863  21.82050477]]]]
 */

    const auto&t = this->top_;
    EXPECT_NEAR(t[0], 3.13084041, 1e-6);
    EXPECT_NEAR(t[1], 3.65058119, 1e-6);
    EXPECT_NEAR(t[2], -2.36481881, 1e-5);
    EXPECT_NEAR(t[3], 8.70304113, 1e-5);
    EXPECT_NEAR(t[4], 2.87097002, 1e-6);
    EXPECT_NEAR(t[5], 2.52447616, 1e-6);
    EXPECT_NEAR(t[6], 11.5724863, 1e-5);
    EXPECT_NEAR(t[7], 21.82050477, 1e-5);
}

TYPED_TEST(BatchNormalizationLayerTest, bprop_with_jet_gamma)
{
    static constexpr int N = 3;
    static constexpr int C = 2;
    static constexpr int H = 5;
    static constexpr int W = 4;
    static constexpr int DIM = C;

    using Type = Jet<TypeParam, DIM>;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(BATCH_NORMALIZATION);
    auto layer = Layer<Type>::create(proto);

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);
    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    for (int i = 0; i < 5; i++)
    {
        gaussian<Type>(layer->mutable_param()[0], 0, 1);
        gaussian<Type>(layer->mutable_param()[1], 0, 1);
        for (int i = 0; i < DIM; i++)
        {
            layer->param()[0]->d_[i].v_[i] = 1;
        }

        uniform<Type>(&bottom, -100, 100);
        uniform<Type>(&top_gradient, -100, 100);
        set_to<Type>(layer->mutable_gradient()[0], 0);

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
            EXPECT_NEAR(layer->gradient()[0]->d_[i].a_ / expected, 1, 1e-5);
        }
    }
}

TYPED_TEST(BatchNormalizationLayerTest, bprop_with_jet_beta)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;
    static constexpr int DIM = C;

    using Type = Jet<TypeParam, DIM>;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(BATCH_NORMALIZATION);
    auto layer = Layer<Type>::create(proto);

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);
    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});


    for (int i = 0; i < 5; i++)
    {
        gaussian<Type>(layer->mutable_param()[0], 0, 1);
        gaussian<Type>(layer->mutable_param()[1], 0, 1);
        for (int i = 0; i < DIM; i++)
        {
            layer->param()[1]->d_[i].v_[i] = 1;
        }

        uniform<Type>(&bottom, -100, 100);
        uniform<Type>(&top_gradient, -100, 100);
        set_to<Type>(layer->mutable_gradient()[1], 0);

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
            EXPECT_NEAR(layer->gradient()[1]->d_[i].a_ / expected, 1, 1e-5);
        }
    }
}

TYPED_TEST(BatchNormalizationLayerTest, bprop_with_jet_input)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;
    static constexpr int DIM = N*C*H*W;

    using Type = Jet<TypeParam, DIM>;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(BATCH_NORMALIZATION);
    auto layer = Layer<Type>::create(proto);

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);
    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    for (int i = 0; i < 8; i++)
    {
        uniform<Type>(&bottom, -100, 100);
        for (int i = 0; i < DIM; i++)
        {
            bottom[i].v_[i] = 1;
        }

        gaussian<Type>(layer->mutable_param()[0], 0, 1);
        gaussian<Type>(layer->mutable_param()[1], 0, 1);

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
            EXPECT_NEAR(bottom_gradient[i].a_ / expected, 1, 1e-4);
        }
    }
}

}  // namespace cnn

