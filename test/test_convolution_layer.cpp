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

TYPED_TEST(ConvolutionLayerTest, fprop)
{
    this->num_output_ = 1;
    this->kernel_size_ = 3;

    LayerProto proto;
    proto.set_phase(TEST);
    proto.set_type(CONVOLUTION);
    proto.mutable_conv_proto()->set_num_output(this->num_output_);
    proto.mutable_conv_proto()->set_kernel_size(this->kernel_size_);

    this->layer_ = Layer<TypeParam>::create(proto);

    const static int N = 1;
    const static int C = 1;
    const static int H = 5;
    const static int W = 5;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    TypeParam d[H*W] = {
       1,2,1,-1,3,
      -1,1,2,-3,2,
       0,2,1,-2,1,
       2,0,-1,1,0,
       0,1,1,-1,1};
    for (int i = 0; i < H*W; i++)
    {
        this->bottom_.d_[i] = d[i];
    }

    TypeParam weight[9] = {
        1, -1, 1,
        0, 1, -1,
        1, 2, 0};

    auto& w = *this->layer_->mutable_param()[0];
    for (int i = 0; i < 9; i++)
    {
        w[i] = weight[i];
    }

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = 10;
    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam expected[H*W] = {
        -3, 2, 7, -8, 4,
        -1, 3, 9, -3, -2,
        4, 3, -3, 5, -3,
        4, 2, 0, 4, -2,
        -3, 1, 4, -4, 2};
    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ(expected[i] + b[0], this->top_[i]);
    }
}

TYPED_TEST(ConvolutionLayerTest, fprop2)
{
    this->num_output_ = 1;
    this->kernel_size_ = 3;

    LayerProto proto;
    proto.set_phase(TEST);
    proto.set_type(CONVOLUTION);
    proto.mutable_conv_proto()->set_num_output(this->num_output_);
    proto.mutable_conv_proto()->set_kernel_size(this->kernel_size_);

    this->layer_ = Layer<TypeParam>::create(proto);

    const static int N = 1;
    const static int C = 1;
    const static int H = 5;
    const static int W = 5;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    TypeParam d[H*W] = {
        1, 2, -1, -2, 3,
        2, 1, 1, -1, -2,
        -1, -2, -1, 1, 2,
        1, 0, 1, -1, 0,
        0, -1, 0, 1, -1
    };
    for (int i = 0; i < H*W; i++)
    {
        this->bottom_.d_[i] = d[i];
    }

    TypeParam weight[9] = {
        1, 0, -1,
        1, 1, 0,
        0, 0, 2,
    };

    auto& w = *this->layer_->mutable_param()[0];
    for (int i = 0; i < 9; i++)
    {
        w[i] = weight[i];
    }

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = -2;
    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam expected[H*W] = {
        3, 5, -1, -7, 1,
        -4, 3, 8, 0, -5,
        -2, 0, -3, 3, 2,
        1, 1, 0, -5, 0,
        0, -1, 0, 2, -1};
    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ(expected[i] + b[0], this->top_[i]);
    }
}

TYPED_TEST(ConvolutionLayerTest, fprop3)
{
    this->num_output_ = 1;
    this->kernel_size_ = 3;

    LayerProto proto;
    proto.set_phase(TEST);
    proto.set_type(CONVOLUTION);
    proto.mutable_conv_proto()->set_num_output(this->num_output_);
    proto.mutable_conv_proto()->set_kernel_size(this->kernel_size_);

    this->layer_ = Layer<TypeParam>::create(proto);

    const static int N = 1;
    const static int C = 2;
    const static int H = 5;
    const static int W = 5;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    TypeParam d[C*H*W] = {
       1,2,1,-1,3,
      -1,1,2,-3,2,
       0,2,1,-2,1,
       2,0,-1,1,0,
       0,1,1,-1,1,

        1, 2, -1, -2, 3,
        2, 1, 1, -1, -2,
        -1, -2, -1, 1, 2,
        1, 0, 1, -1, 0,
        0, -1, 0, 1, -1
    };
    for (int i = 0; i < C*H*W; i++)
    {
        this->bottom_.d_[i] = d[i];
    }

    TypeParam weight[18] = {
        1, -1, 1,
        0, 1, -1,
        1, 2, 0,

        1, 0, -1,
        1, 1, 0,
        0, 0, 2,
    };

    auto& w = *this->layer_->mutable_param()[0];
    for (int i = 0; i < 18; i++)
    {
        w[i] = weight[i];
    }

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = 10;
    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam expected1[H*W] = {
        -3, 2, 7, -8, 4,
        -1, 3, 9, -3, -2,
        4, 3, -3, 5, -3,
        4, 2, 0, 4, -2,
        -3, 1, 4, -4, 2};

    TypeParam expected2[H*W] = {
        3, 5, -1, -7, 1,
        -4, 3, 8, 0, -5,
        -2, 0, -3, 3, 2,
        1, 1, 0, -5, 0,
        0, -1, 0, 2, -1};

    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ(expected1[i] + expected2[i] + b[0], this->top_[i]);
    }
}

TYPED_TEST(ConvolutionLayerTest, fprop4)
{
    this->num_output_ = 1;
    this->kernel_size_ = 3;

    LayerProto proto;
    proto.set_phase(TEST);
    proto.set_type(CONVOLUTION);
    proto.mutable_conv_proto()->set_num_output(this->num_output_);
    proto.mutable_conv_proto()->set_kernel_size(this->kernel_size_);

    this->layer_ = Layer<TypeParam>::create(proto);

    const static int N = 2;
    const static int C = 2;
    const static int H = 5;
    const static int W = 5;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    TypeParam d[C*H*W] = {
       1,2,1,-1,3,
      -1,1,2,-3,2,
       0,2,1,-2,1,
       2,0,-1,1,0,
       0,1,1,-1,1,

        1, 2, -1, -2, 3,
        2, 1, 1, -1, -2,
        -1, -2, -1, 1, 2,
        1, 0, 1, -1, 0,
        0, -1, 0, 1, -1
    };
    for (int i = 0; i < C*H*W; i++)
    {
        this->bottom_.d_[i] = d[i];
    }

    for (int i = 0; i < C*H*W; i++)
    {
        this->bottom_.d_[i+C*H*W] = d[i] * 2;
    }

    TypeParam weight[18] = {
        1, -1, 1,
        0, 1, -1,
        1, 2, 0,

        1, 0, -1,
        1, 1, 0,
        0, 0, 2,
    };

    auto& w = *this->layer_->mutable_param()[0];
    for (int i = 0; i < 18; i++)
    {
        w[i] = weight[i];
    }

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = 10;
    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam expected1[H*W] = {
        -3, 2, 7, -8, 4,
        -1, 3, 9, -3, -2,
        4, 3, -3, 5, -3,
        4, 2, 0, 4, -2,
        -3, 1, 4, -4, 2};

    TypeParam expected2[H*W] = {
        3, 5, -1, -7, 1,
        -4, 3, 8, 0, -5,
        -2, 0, -3, 3, 2,
        1, 1, 0, -5, 0,
        0, -1, 0, 2, -1};

    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ(expected1[i] + expected2[i] + b[0], this->top_[i]);
    }

    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ((expected1[i] + expected2[i])*2 + b[0], this->top_[i+H*W]);
    }
}

TYPED_TEST(ConvolutionLayerTest, fprop5)
{
    this->num_output_ = 2;
    this->kernel_size_ = 3;

    LayerProto proto;
    proto.set_phase(TEST);
    proto.set_type(CONVOLUTION);
    proto.mutable_conv_proto()->set_num_output(this->num_output_);
    proto.mutable_conv_proto()->set_kernel_size(this->kernel_size_);

    this->layer_ = Layer<TypeParam>::create(proto);

    const static int N = 2;
    const static int C = 2;
    const static int H = 5;
    const static int W = 5;

    this->bottom_.init(N, C, H, W);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    TypeParam d[C*H*W] = {
       1,2,1,-1,3,
      -1,1,2,-3,2,
       0,2,1,-2,1,
       2,0,-1,1,0,
       0,1,1,-1,1,

        1, 2, -1, -2, 3,
        2, 1, 1, -1, -2,
        -1, -2, -1, 1, 2,
        1, 0, 1, -1, 0,
        0, -1, 0, 1, -1
    };

    // batch 0, channel 0&1
    for (int i = 0; i < C*H*W; i++)
    {
        this->bottom_.d_[i] = d[i];
    }

    // batch 1, channel 0&1
    for (int i = 0; i < C*H*W; i++)
    {
        this->bottom_.d_[i+C*H*W] = d[i] * 4;
    }

    TypeParam weight[18] = {
        1, -1, 1,
        0, 1, -1,
        1, 2, 0,

        1, 0, -1,
        1, 1, 0,
        0, 0, 2,
    };

    // for output 0
    auto& w = *this->layer_->mutable_param()[0];
    for (int i = 0; i < 18; i++)
    {
        w[i] = weight[i];
    }

    // for output 1
    for (int i = 0; i < 18; i++)
    {
        w[18 + i] = weight[i] / 2;
    }

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = 10;
    b[1] = -5;
    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam expected1[H*W] = {
        -3, 2, 7, -8, 4,
        -1, 3, 9, -3, -2,
        4, 3, -3, 5, -3,
        4, 2, 0, 4, -2,
        -3, 1, 4, -4, 2};

    TypeParam expected2[H*W] = {
        3, 5, -1, -7, 1,
        -4, 3, 8, 0, -5,
        -2, 0, -3, 3, 2,
        1, 1, 0, -5, 0,
        0, -1, 0, 2, -1};

    // batch 0 output 0
    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ(expected1[i] + expected2[i] + b[0], this->top_[i]);
    }

    // batch 0, output 1
    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ((expected1[i] + expected2[i])/2 + b[1], this->top_[i+H*W]);
    }

    // batch 1 output 0
    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ((expected1[i] + expected2[i])*4 + b[0], this->top_[i + 2*H*W]);
    }

    // batch 1 output 1
    for (int i = 0; i < H*W; i++)
    {
        EXPECT_EQ((expected1[i] + expected2[i])*2 + b[1], this->top_[i + 3*H*W]);
    }
}

}  // namespace cnn
