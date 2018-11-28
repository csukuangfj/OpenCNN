#include <gtest/gtest.h>
#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class LayerTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LayerTest, MyTypes);

TYPED_TEST(LayerTest, create_fc)
{
    LayerProto proto;
    proto.set_type(FULL_CONNECTED);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_input)
{
    LayerProto proto;
    proto.set_type(INPUT);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_l2_loss)
{
    LayerProto proto;
    proto.set_type(L2_LOSS);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_softmax)
{
    LayerProto proto;
    proto.set_type(SOFTMAX);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_log_loss)
{
    LayerProto proto;
    proto.set_type(LOG_LOSS);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_softmax_with_log_loss)
{
    LayerProto proto;
    proto.set_type(SOFTMAX_WITH_LOG_LOSS);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_convolution)
{
    LayerProto proto;
    proto.set_type(CONVOLUTION);
    auto* p = proto.mutable_conv_proto();
    p->set_num_output(2);
    p->set_kernel_size(1);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_relu)
{
    LayerProto proto;
    proto.set_type(RELU);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_max_pooling)
{
    LayerProto proto;
    proto.set_type(MAX_POOLING);
    auto* p = proto.mutable_max_pooling_proto();
    p->set_win_size(2);
    p->set_stride(2);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_dropout)
{
    LayerProto proto;
    proto.set_type(DROP_OUT);
    auto* p = proto.mutable_dropout_proto();
    p->set_keep_prob(0.8);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_batch_normalization)
{
    LayerProto proto;
    proto.set_type(BATCH_NORMALIZATION);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

TYPED_TEST(LayerTest, create_leaky_relu)
{
    LayerProto proto;
    proto.set_type(LEAKY_RELU);
    proto.mutable_leaky_relu_proto()->set_alpha(0.3);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
}

}  // namespace cnn

