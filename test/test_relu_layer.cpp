#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/array_math.hpp"
#include "cnn/layer.hpp"
#include "cnn/jet.hpp"

namespace cnn
{

template<typename Dtype>
class ReLULayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        proto.set_type(RELU);
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
TYPED_TEST_CASE(ReLULayerTest, MyTypes);

TYPED_TEST(ReLULayerTest, reshape_train_phase)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;

    this->layer_->proto().set_phase(TRAIN);

    this->bottom_.init(N, C, H, W);
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

TYPED_TEST(ReLULayerTest, reshape_test_phase)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;

    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(N, C, H, W);
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

TYPED_TEST(ReLULayerTest, fprop)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;

    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(N, C, H, W);
    uniform<TypeParam>(&this->bottom_, -100, 100);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});

    this->layer_->fprop(
            {&this->bottom_},
            {&this->top_});

    TypeParam expected;

    const auto& b = this->bottom_;
    const auto& t = this->top_;

    for (int i = 0; i < b.total_; i++)
    {
        expected = (b[i] >= 0) ? b[i] : 0;
        EXPECT_EQ(t[i], expected);
    }
}

TYPED_TEST(ReLULayerTest, bprop_with_jet)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;
    static constexpr int DIM = 1;

    using Type = Jet<TypeParam, DIM>;

    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(RELU);
    auto layer = Layer<Type>::create(proto);

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);

    uniform<Type>(&bottom, -100, 100);
    for (int i = 0; i < bottom.total_; i++)
    {
        bottom.d_[i].v_[0] = 1;
    }

    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    uniform<Type>(&top_gradient, -100, 100);
    layer->fprop({&bottom}, {&top});
    layer->bprop(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    for (int i = 0; i < bottom_gradient.total_; i++)
    {
        TypeParam expected = top[i].v_[0]*top_gradient[i].a_;
        EXPECT_EQ(bottom_gradient[i], expected);
    }
}

}  // namespace cnn
