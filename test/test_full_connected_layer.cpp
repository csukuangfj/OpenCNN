#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/array_math.hpp"
#include "cnn/layer.hpp"
#include "cnn/jet.hpp"

namespace cnn
{

template<typename Dtype>
class FullConnectedLayerTest : public ::testing::Test
{
    void SetUp() override
    {
        LayerProto proto;
        num_output_ = 10;
        proto.set_type(FULL_CONNECTED);
        proto.mutable_fc_proto()->set_num_output(num_output_);
        layer_ = Layer<Dtype>::create(proto);
    }
protected:
    std::shared_ptr<Layer<Dtype>> layer_;

    Array<Dtype> bottom_;
    Array<Dtype> bottom_gradient_;
    Array<Dtype> top_;
    Array<Dtype> top_gradient_;

    int num_output_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FullConnectedLayerTest, MyTypes);

TYPED_TEST(FullConnectedLayerTest, reshape_train_phase)
{
    this->layer_->proto().set_phase(TRAIN);

    this->bottom_.init(2, 3, 4, 5);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_.has_same_shape({2, this->num_output_, 1, 1}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(
                {2, this->num_output_, 1, 1}));

    EXPECT_TRUE(this->layer_->param()[0]->has_same_shape(
                {1, 1, this->num_output_, 3*4*5}));
    EXPECT_TRUE(this->layer_->param()[1]->has_same_shape(
                {1, 1, 1, this->num_output_}));

    EXPECT_TRUE(this->layer_->gradient()[0]->has_same_shape(
                {1, 1, this->num_output_, 3*4*5}));
    EXPECT_TRUE(this->layer_->gradient()[1]->has_same_shape(
                {1, 1, 1, this->num_output_}));
}

TYPED_TEST(FullConnectedLayerTest, reshape_test_phase)
{
    this->layer_->proto().set_phase(TEST);

    this->bottom_.init(3, 6, 8, 5);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});
    EXPECT_TRUE(this->bottom_gradient_.has_same_shape({0, 0, 0, 0}));
    EXPECT_TRUE(this->top_.has_same_shape({3, this->num_output_, 1, 1}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(
                {0, 0, 0, 0}));

    EXPECT_TRUE(this->layer_->param()[0]->has_same_shape(
                {1, 1, this->num_output_, 6*8*5}));
    EXPECT_TRUE(this->layer_->param()[1]->has_same_shape(
                {1, 1, 1, this->num_output_}));

    EXPECT_TRUE(this->layer_->gradient().empty());
}

TYPED_TEST(FullConnectedLayerTest, fprop)
{
    this->num_output_ = 2;
    LayerProto proto;
    proto.set_phase(TEST);
    proto.set_type(FULL_CONNECTED);
    proto.mutable_fc_proto()->set_num_output(this->num_output_);

    this->layer_ = Layer<TypeParam>::create(proto);

    this->bottom_.init(2, 3, 1, 2);
    this->layer_->reshape(
            {&this->bottom_},
            {},
            {&this->top_},
            {});
    /*
     * first batch
     * 1 -1
     * 2 -3
     * 1  2
     *
     * second batch
     *
     * 1  2
     * -5 4
     * 2  1
     */
    this->bottom_(0, 0, 0, 0) = 1;
    this->bottom_(0, 0, 0, 1) = -1;
    this->bottom_(0, 1, 0, 0) = 2;
    this->bottom_(0, 1, 0, 1) = -3;
    this->bottom_(0, 2, 0, 0) = 1;
    this->bottom_(0, 2, 0, 1) = 2;

    this->bottom_(1, 0, 0, 0) = 1;
    this->bottom_(1, 0, 0, 1) = 2;
    this->bottom_(1, 1, 0, 0) = -5;
    this->bottom_(1, 1, 0, 1) = 4;
    this->bottom_(1, 2, 0, 0) = 2;
    this->bottom_(1, 2, 0, 1) = 1;

    /*
     * set weights
     *
     * 1 1 -1 0 2 3
     * -1 1 0 0 1 2
     *
     * set bias
     * 10
     * 2
     */
    auto& w = *this->layer_->mutable_param()[0];
    w[0] = 1; w[1] = 1; w[2] = -1; w[3] = 0; w[4] = 2; w[5] = 3;
    w[6] = -1; w[7] = 1; w[8] = 0; w[9] = 0; w[10] = 1; w[11] = 2;

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = 10; b[1] = 2;

    this->layer_->fprop({&this->bottom_}, {&this->top_});

    TypeParam expected00;
    TypeParam expected01;

    TypeParam expected10;
    TypeParam expected11;

    expected00 = ax_dot_by<TypeParam>(6, 1, &w[0], 1, &this->bottom_[0]);
    expected00 += b[0];

    expected01 = ax_dot_by<TypeParam>(6, 1, &w[6], 1, &this->bottom_[0]);
    expected01 += b[1];

    expected10 = ax_dot_by<TypeParam>(6, 1, &w[0], 1, &this->bottom_[6]);
    expected10 += b[0];

    expected11 = ax_dot_by<TypeParam>(6, 1, &w[6], 1, &this->bottom_[6]);
    expected11 += b[1];

    EXPECT_EQ(expected00, this->top_[0]);
    EXPECT_EQ(expected01, this->top_[1]);

    EXPECT_EQ(expected10, this->top_[2]);
    EXPECT_EQ(expected11, this->top_[3]);
}

TYPED_TEST(FullConnectedLayerTest, bprop_with_jet_input)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 5;
    static constexpr int W = 4;
    static constexpr int DIM = C*H*W;

    using Type = Jet<TypeParam, DIM>;

    this->num_output_ = 6;
    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(FULL_CONNECTED);
    proto.mutable_fc_proto()->set_num_output(this->num_output_);
    auto layer = Layer<Type>::create(proto);

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);

    uniform<Type>(&bottom, -100, 100);
    for (int n = 0; n < N; n++)
    for (int i = 0; i < DIM; i++)
    {
        bottom.d_[n*DIM + i].v_[i] = 1;
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

    for (int n = 0; n < N; n++)
    for (int i = 0; i < DIM; i++)
    {
        TypeParam expected = 0;
        for (int k = 0; k < this->num_output_; k++)
        {
            expected += top[n*this->num_output_ + k].v_[i] *
                top_gradient[n*this->num_output_ + k].a_;
        }
        EXPECT_NEAR(bottom_gradient[n*DIM + i], expected, 1e-5);
    }
}

TYPED_TEST(FullConnectedLayerTest, bprop_with_jet)
{
    static constexpr int N = 2;
    static constexpr int C = 3;
    static constexpr int H = 4;
    static constexpr int W = 5;
    static constexpr int DIM = C*H*W;

    this->num_output_ = 6;
    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(FULL_CONNECTED);
    proto.mutable_fc_proto()->set_num_output(this->num_output_);
    auto layer = Layer<Jet<TypeParam, DIM>>::create(proto);

    using Type = Jet<TypeParam, DIM>;

    Array<Type> bottom;
    Array<Type> bottom_gradient;
    Array<Type> top;
    Array<Type> top_gradient;

    bottom.init(N, C, H, W);

    uniform<Type>(&bottom, -100, 100);

    layer->reshape(
            {&bottom},
            {&bottom_gradient},
            {&top},
            {&top_gradient});

    uniform<Type>(&top_gradient, -100, 100);

    // test each row of the weight matrix
    for (int k = 0; k < this->num_output_; k++)
    {
        auto& w = *layer->mutable_param()[0];
        auto &dw = *layer->mutable_gradient()[0];
        set_to<Type>(&dw, 0);

        gaussian<Type>(&w, 0, 1);

        for (int i = 0; i < DIM; i++)
        {
            w[k*DIM + i].v_[i] = 1;
        }

        layer->fprop({&bottom}, {&top});
        layer->bprop(
                {&bottom},
                {&bottom_gradient},
                {&top},
                {&top_gradient});

        for (int i = 0; i < DIM; i++)
        {
            TypeParam expected = 0;
            for (int n = 0; n < N; n++)
            {

                expected += top[n*this->num_output_ + k].v_[i] *
                    top_gradient[n*this->num_output_ + k].a_;

            }
            EXPECT_NEAR(dw[k*DIM + i].a_, expected, 1e-5);
        }
    }


    // we have to clear the effect of the weight matrix
    // since it occupies the same dual number space with
    // the bias
    auto& w = *layer->mutable_param()[0];
    gaussian<Type>(&w, 0, 1);

    // test each row of the bias
    for (int k = 0; k < this->num_output_; k++)
    {
        auto& b = *layer->mutable_param()[1];

        auto &db = *layer->mutable_gradient()[1];
        set_to<Type>(&db, 0);

        gaussian<Type>(&b, 0, 1);

        b[k].v_[0] = 1;

        layer->fprop({&bottom}, {&top});
        layer->bprop(
                {&bottom},
                {&bottom_gradient},
                {&top},
                {&top_gradient});

        TypeParam expected = 0;
        for (int n = 0; n < N; n++)
        {

            expected += top[n*this->num_output_ + k].v_[0] *
                top_gradient[n*this->num_output_ + k].a_;

        }
        EXPECT_NEAR(db[k].a_, expected, 1e-5);
    }
}

}  // namespace cnn

