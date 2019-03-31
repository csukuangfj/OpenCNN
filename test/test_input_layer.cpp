
#include <gtest/gtest.h>

#define private public

#include "cnn/layer.hpp"

namespace cnn {

template <typename Dtype>
class InputLayerTest : public ::testing::Test {};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(InputLayerTest, MyTypes);

TYPED_TEST(InputLayerTest, create) {
  LayerProto proto;
  proto.set_type(INPUT);

  auto param = proto.mutable_input_proto();
  param->set_n(2);
  param->set_c(3);
  param->set_h(4);
  param->set_w(5);

  InputLayer<TypeParam> layer(proto);
  EXPECT_EQ(layer.n_, 2);
  EXPECT_EQ(layer.c_, 3);
  EXPECT_EQ(layer.h_, 4);
  EXPECT_EQ(layer.w_, 5);

  Array<TypeParam> arr;
  std::vector<Array<TypeParam>*> vec{&arr};
  layer.reshape({}, {}, vec, {});

  EXPECT_EQ(arr.n_, 2);
  EXPECT_EQ(arr.c_, 3);
  EXPECT_EQ(arr.h_, 4);
  EXPECT_EQ(arr.w_, 5);
}

}  // namespace cnn
