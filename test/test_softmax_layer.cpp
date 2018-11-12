#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/softmax_layer.hpp"

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

    int n = 2;  // batch size
    this->bottom_.init(n, 3, 4, 5);
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

    int n = 3;  // batch size
    this->bottom_.init(n, 9, 6, 8);
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
     */
    auto& d = this->bottom_;
    d[0] = 6;
    d[1] = -1;
    d[2] = 10;
    d[3] = 2;
    d[4] = 5;
    d[5] = 1;

    this->layer_->fprop(
            {&this->bottom_},
            {&this->top_});

    auto& t = this->top_;
    EXPECT_NEAR(t[0], 9.82013790e-01, 1e-5);
    EXPECT_NEAR(t[1], 2.47262316e-03, 1e-5);
    EXPECT_NEAR(t[2], 9.99876605e-01, 1e-5);
    EXPECT_NEAR(t[3], 1.79862100e-02, 1e-5);
    EXPECT_NEAR(t[4], 9.97527377e-01, 1e-5);
    EXPECT_NEAR(t[5], 1.23394576e-04, 1e-5);
}

}  // namespace cnn


