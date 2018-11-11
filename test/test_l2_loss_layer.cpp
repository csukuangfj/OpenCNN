#include <gtest/gtest.h>
#include <stdlib.h>     // for rand()

#include "cnn/array_math.hpp"
#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class L2LossLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        proto.set_type(L2_LOSS);
        l2_loss_layer_ = Layer<Dtype>::create(proto);
    }
protected:
    std::shared_ptr<Layer<Dtype>> l2_loss_layer_;

};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(L2LossLayerTest, MyTypes);

TYPED_TEST(L2LossLayerTest, fprop)
{
    Array<TypeParam> arr1;
    Array<TypeParam> arr2;
    arr1.init(2, 3, 4, 5);
    arr2.init(2, 3, 4, 5);

    set_to<TypeParam>(&arr1, 2);
    set_to<TypeParam>(&arr2, 2);

    int n = rand()%arr1.total_;
    for (int i = 0; i < n; i++)
    {
        arr2[i] = 1;
    }

    Array<TypeParam> out;

    this->l2_loss_layer_->reshape({&arr1, &arr2}, {&out});
    this->l2_loss_layer_->fprop({&arr1, &arr2}, {&out});
    EXPECT_EQ(out[0], n);
}

}  // namespace cnn


