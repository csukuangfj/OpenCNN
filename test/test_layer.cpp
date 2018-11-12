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

}  // namespace cnn

