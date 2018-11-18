#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class ConvolutionLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        num_output_ = 2;
        kernel_size_ = 3;
        proto.set_type(CONVOLUTION);
        proto.mutable_conv_proto()->set_num_output(num_output_);
        proto.mutable_conv_proto()->set_kernel_size(kernel_size_);
        layer_ = Layer<Dtype>::create(proto);
    }
protected:
    std::shared_ptr<Layer<Dtype>> layer_;

    Array<Dtype> bottom_;
    Array<Dtype> bottom_gradient_;
    Array<Dtype> top_;
    Array<Dtype> top_gradient_;

    int num_output_;
    int kernel_size_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ConvolutionLayerTest, MyTypes);

TYPED_TEST(ConvolutionLayerTest, reshape_train_phase)
{
    const static int N = 2;
    const static int C = 3;
    const static int H = 4;
    const static int W = 5;

    this->layer_->proto().set_phase(TRAIN);

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_.has_same_shape(
                {N, this->num_output_, H, W}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(
                {N, this->num_output_, H, W}));

    EXPECT_EQ(this->layer_->param().size(), 2);
    EXPECT_TRUE(this->layer_->param()[0]->has_same_shape(
                {this->num_output_,
                C,
                this->kernel_size_,
                this->kernel_size_}));

    EXPECT_TRUE(this->layer_->param()[1]->has_same_shape(
                {1, 1, 1, this->num_output_}));

    EXPECT_EQ(this->layer_->gradient().size(), 2);
    EXPECT_TRUE(this->layer_->gradient()[0]->has_same_shape(
                {this->num_output_,
                 C,
                 this->kernel_size_,
                 this->kernel_size_}));
    EXPECT_TRUE(this->layer_->gradient()[1]->has_same_shape(
                {1, 1, 1, this->num_output_}));
}

TYPED_TEST(ConvolutionLayerTest, reshape_test_phase)
{
    const static int N = 2;
    const static int C = 3;
    const static int H = 4;
    const static int W = 5;

    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape({0, 0, 0, 0}));
    EXPECT_TRUE(this->top_.has_same_shape(
                {N, this->num_output_, H, W}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape({0, 0, 0, 0}));

    EXPECT_EQ(this->layer_->param().size(), 2);
    EXPECT_TRUE(this->layer_->param()[0]->has_same_shape(
                {this->num_output_,
                C,
                this->kernel_size_,
                this->kernel_size_}));

    EXPECT_TRUE(this->layer_->param()[1]->has_same_shape(
                {1, 1, 1, this->num_output_}));

    EXPECT_EQ(this->layer_->gradient().size(), 0);
}

}  // namespace cnn
