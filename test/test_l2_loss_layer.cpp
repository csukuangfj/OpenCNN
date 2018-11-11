#include <glog/logging.h>
#include <gtest/gtest.h>

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

    int n = uniform(1, arr1.total_);
    for (int i = 0; i < n; i++)
    {
        arr2[i] = 1;
    }

    Array<TypeParam> out;

    this->l2_loss_layer_->reshape({&arr1, &arr2}, {&out});
    this->l2_loss_layer_->fprop({&arr1, &arr2}, {&out});
    EXPECT_NEAR(out[0], TypeParam(n)/arr1.total_, 1e-4);
}

// TODO(fangjun): add gradient checker
TYPED_TEST(L2LossLayerTest, bprop)
{
    Array<TypeParam> arr1;
    Array<TypeParam> arr2;
    arr1.init(2, 5, 3, 4);
    arr2.init(2, 5, 3, 4);

    set_to<TypeParam>(&arr1, 2);
    set_to<TypeParam>(&arr2, 2);

    int n = uniform(1, arr1.total_-1);
    for (int i = 0; i < n; i++)
    {
        arr2[i] = 1;
    }

    Array<TypeParam> out;

    this->l2_loss_layer_->reshape({&arr1, &arr2}, {&out});
    this->l2_loss_layer_->fprop({&arr1, &arr2}, {&out});

    Array<TypeParam> gradient;
    gradient.init_like(arr1);
    this->l2_loss_layer_->bprop(
            {&arr1, &arr2},
            {&gradient},
            {&out},
            {});
    for (int i = 0; i < gradient.total_; i++)
    {
        if (i < n)
        {
            EXPECT_EQ(gradient[i], 1);
        }
        else
        {
            EXPECT_EQ(gradient[i], 0);
        }
    }
}


}  // namespace cnn


