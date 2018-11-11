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
    auto fc_proto = proto.mutable_fc_proto();
    fc_proto->set_num_output(10);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
    res->fprop({}, nullptr);
}

TYPED_TEST(LayerTest, create_input)
{
    LayerProto proto;
    proto.set_type(INPUT);
    auto res = Layer<TypeParam>::create(proto);
    EXPECT_NE(res.get(), nullptr);
    res->fprop({}, nullptr);
}

}  // namespace cnn

