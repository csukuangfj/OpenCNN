#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class SoftmaxLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        proto.set_type(SOFTMAX);
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
TYPED_TEST_CASE(SoftmaxLayerTest, MyTypes);

TYPED_TEST(SoftmaxLayerTest, reshape_train_phase)
{
    this->layer_->proto().set_phase(TRAIN);

    this->bottom_.init(2, 3, 4, 5);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(this->bottom_));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(SoftmaxLayerTest, reshape_test_phase)
{
    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(3, 9, 6, 8);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape({0, 0, 0, 0}));
    EXPECT_TRUE(this->top_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_TRUE(this->layer_->param().empty());
    EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(SoftmaxLayerTest, fprop)
{
    this->layer_->proto().set_phase(TEST);
    this->bottom_.init(2, 2, 1, 3);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});
    /*
     * batch 0
     * 6 -1 10
     * 2  5 1
>>> import numpy as np
>>> a = np.array([[6, -1, 10], [2, 5, 1]])
>>> b = np.max(a, axis=0)
>>> a
array([[ 6, -1, 10],
       [ 2,  5,  1]])
>>> b
array([ 6,  5, 10])
>>> c = a - b
>>> c
array([[ 0, -6,  0],
       [-4,  0, -9]])
>>> d = np.exp(c)
>>> d
array([[1.00000000e+00, 2.47875218e-03, 1.00000000e+00],
       [1.83156389e-02, 1.00000000e+00, 1.23409804e-04]])
>>> f = np.sum(d, axis=0)
>>> f
array([1.01831564, 1.00247875, 1.00012341])
>>> d/f
array([[9.82013790e-01, 2.47262316e-03, 9.99876605e-01],
       [1.79862100e-02, 9.97527377e-01, 1.23394576e-04]])

----------------------------------------
    **batch1**
----------------------------------------

1  -2  5
6  9   3

>>> import numpy as np
>>> a = np.array([[1, -2, 5], [6, 9, 3]])
>>> a
array([[ 1, -2,  5],
       [ 6,  9,  3]])
>>> b = np.max(a, axis = 0)
>>> b
array([6, 9, 5])
>>> c = a - b
>>> c
array([[ -5, -11,   0],
       [  0,   0,  -2]])
>>> d = np.exp(c)
>>> d
array([[6.73794700e-03, 1.67017008e-05, 1.00000000e+00],
       [1.00000000e+00, 1.00000000e+00, 1.35335283e-01]])
>>> f = np.sum(d, axis = 0)
>>> f
array([1.00673795, 1.0000167 , 1.13533528])
>>> d/f
array([[6.69285092e-03, 1.67014218e-05, 8.80797078e-01],
       [9.93307149e-01, 9.99983299e-01, 1.19202922e-01]])

     */
    auto& d = this->bottom_;
    d[0] = 6;
    d[1] = -1;
    d[2] = 10;
    d[3] = 2;
    d[4] = 5;
    d[5] = 1;

    d[6] = 1;
    d[7] = -2;
    d[8] = 5;
    d[9] = 6;
    d[10] = 9;
    d[11] = 3;

    this->layer_->fprop(
            {&this->bottom_},
            {&this->top_});

    auto& t = this->top_;
    // batch 0
    EXPECT_NEAR(t[0], 9.82013790e-01, 1e-5);
    EXPECT_NEAR(t[1], 2.47262316e-03, 1e-5);
    EXPECT_NEAR(t[2], 9.99876605e-01, 1e-5);
    EXPECT_NEAR(t[3], 1.79862100e-02, 1e-5);
    EXPECT_NEAR(t[4], 9.97527377e-01, 1e-5);
    EXPECT_NEAR(t[5], 1.23394576e-04, 1e-5);

    // batch 1
    EXPECT_NEAR(t[6], 6.69285092e-03, 1e-5);
    EXPECT_NEAR(t[7], 1.67014218e-05, 1e-5);
    EXPECT_NEAR(t[8], 8.80797078e-01, 1e-5);
    EXPECT_NEAR(t[9], 9.93307149e-01, 1e-5);
    EXPECT_NEAR(t[10], 9.99983299e-01, 1e-5);
    EXPECT_NEAR(t[11], 1.19202922e-01, 1e-5);
}

TYPED_TEST(SoftmaxLayerTest, bprop_with_jet)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 3;
    static constexpr int W = 4;
    static constexpr int DIM = C*H*W;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(SOFTMAX);
    auto layer = Layer<Jet<TypeParam, DIM>>::create(proto);

    using Type = Jet<TypeParam, DIM>;

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);
    uniform<Type>(&bottom, 1, 15);
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C; c++)
    {
        // different pixels can share the same dual number space!
        bottom(n, c, h, w).v_[c] = 1;
    }

    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    layer->fprop({&bottom}, {&top});

    uniform<Type>(&top_gradient, -10, 10);

    layer->bprop(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C; c++)
    {
        TypeParam expected = 0;
        for (int i = 0; i < C; i++)
        {
            expected += top(n, i, h, w).v_[c] * top_gradient(n, i, h, w).a_;
        }
        EXPECT_NEAR(bottom_gradient(n, c, h, w).a_, expected, 1e-5);
    }
}

}  // namespace cnn

