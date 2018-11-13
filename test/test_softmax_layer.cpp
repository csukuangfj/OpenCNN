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

TYPED_TEST(SoftmaxLayerTest, bprop)
{
    this->layer_->proto().set_phase(TRAIN);
    this->bottom_.init(2, 2, 1, 3);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    /*
     * 6 -1 10
     * 2  5  1
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

    auto& tg = this->top_gradient_;
    tg[0] = 1;
    tg[1] = 2;
    tg[2] = 3;
    tg[3] = 4;
    tg[4] = 5;
    tg[5] = 6;

    tg[6] = 7;
    tg[7] = 8;
    tg[8] = 9;
    tg[9] = 0;
    tg[10] = -1;
    tg[11] = -2.5;

    this->layer_->bprop(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});
    const auto& bg = this->bottom_gradient_;
    LOG(INFO) << "bg[6]: " << bg[6];
    LOG(INFO) << "bg[7]: " << bg[7];
    LOG(INFO) << "bg[8]: " << bg[8];
    LOG(INFO) << "bg[9]: " << bg[9];
    LOG(INFO) << "bg[10]: " << bg[10];
    LOG(INFO) << "bg[11]: " << bg[11];

    const auto& t = this->top_;
    TypeParam b;

    b = t(0, 0, 0, 0)*(1 - t(0, 0, 0, 0)) * tg(0, 0, 0, 0)
        + (- t(0, 0, 0, 0)*t(0, 1, 0, 0)) * tg(0, 1, 0, 0);
    EXPECT_NEAR(b, bg[0], 1e-5);    // -0.0529881

    b = t(0, 0, 0, 1)*(1 - t(0, 0, 0, 1)) * tg(0, 0, 0, 1)
        + (- t(0, 0, 0, 1)*t(0, 1, 0, 1)) * tg(0, 1, 0, 1);
    EXPECT_NEAR(b, bg[1], 1e-5);    // -0.00739953

    b = t(0, 0, 0, 2)*(1 - t(0, 0, 0, 2)) * tg(0, 0, 0, 2)
        + (- t(0, 0, 0, 2)*t(0, 1, 0, 2)) * tg(0, 1, 0, 2);
    EXPECT_NEAR(b, bg[2], 1e-5);    // -0.000370138

    b = (- t(0, 0, 0, 0)*t(0, 1, 0, 0)) * tg(0, 0, 0, 0)
        + t(0, 1, 0, 0)*(1 - t(0, 1, 0, 0)) * tg(0, 1, 0, 0);
    EXPECT_NEAR(b, bg[3], 1e-5); // 0.0529881

    b = (- t(0, 0, 0, 1)*t(0, 1, 0, 1)) * tg(0, 0, 0, 1)
        + t(0, 1, 0, 1)*(1 - t(0, 1, 0, 1)) * tg(0, 1, 0, 1);
    EXPECT_NEAR(b, bg[4], 1e-5); // 0.00739953

    b = (- t(0, 0, 0, 2)*t(0, 1, 0, 2)) * tg(0, 0, 0, 2)
        + t(0, 1, 0, 2)*(1 - t(0, 1, 0, 2)) * tg(0, 1, 0, 2);
    EXPECT_NEAR(b, bg[5], 1e-5); // 0.000370138

    // batch 2
    b = t(1, 0, 0, 0)*(1 - t(1, 0, 0, 0)) * tg(1, 0, 0, 0)
        + (- t(1, 0, 0, 0)*t(1, 1, 0, 0)) * tg(1, 1, 0, 0);
    EXPECT_NEAR(b, bg[6], 1e-5);    // 0.0465364

    b = t(1, 0, 0, 1)*(1 - t(1, 0, 0, 1)) * tg(1, 0, 0, 1)
        + (- t(1, 0, 0, 1)*t(1, 1, 0, 1)) * tg(1, 1, 0, 1);
    EXPECT_NEAR(b, bg[7], 1e-5);    // 0.00015031

    b = t(1, 0, 0, 2)*(1 - t(1, 0, 0, 2)) * tg(1, 0, 0, 2)
        + (- t(1, 0, 0, 2)*t(1, 1, 0, 2)) * tg(1, 1, 0, 2);
    EXPECT_NEAR(b, bg[8], 1e-5);    // 1.20743

    b = (- t(1, 0, 0, 0)*t(1, 1, 0, 0)) * tg(1, 0, 0, 0)
        + t(1, 1, 0, 0)*(1 - t(1, 1, 0, 0)) * tg(1, 1, 0, 0);
    EXPECT_NEAR(b, bg[9], 1e-5); // -0.0465364

    b = (- t(1, 0, 0, 1)*t(1, 1, 0, 1)) * tg(1, 0, 0, 1)
        + t(1, 1, 0, 1)*(1 - t(1, 1, 0, 1)) * tg(1, 1, 0, 1);
    EXPECT_NEAR(b, bg[10], 1e-5); // -0.00015031

    b = (- t(1, 0, 0, 2)*t(1, 1, 0, 2)) * tg(1, 0, 0, 2)
        + t(1, 1, 0, 2)*(1 - t(1, 1, 0, 2)) * tg(1, 1, 0, 2);
    EXPECT_NEAR(b, bg[11], 1e-5); // -1.20743
}

}  // namespace cnn


