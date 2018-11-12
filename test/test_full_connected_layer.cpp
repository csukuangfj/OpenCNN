#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/array_math.hpp"
#include "cnn/full_connected_layer.hpp"

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

    int n = 2;  // batch size
    this->bottom_.init(n, 3, 4, 5);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    EXPECT_TRUE(this->bottom_gradient_.has_same_shape(this->bottom_));
    EXPECT_TRUE(this->top_.has_same_shape({n, this->num_output_, 1, 1}));
    EXPECT_TRUE(this->top_gradient_.has_same_shape(
                {n, 1, 1, this->num_output_}));

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

TYPED_TEST(FullConnectedLayerTest, bprop)
{
    this->num_output_ = 2;
    LayerProto proto;
    proto.set_phase(TRAIN);
    proto.set_type(FULL_CONNECTED);
    proto.mutable_fc_proto()->set_num_output(this->num_output_);

    this->layer_ = Layer<TypeParam>::create(proto);

    this->bottom_.init(2, 1, 1, 3);
    this->layer_->reshape(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    // TODO(fangjun)
    // Implement a gradient checker!
    // The following small numbers are only for test purposes.
    // I have manually verified the gradient.

    this->bottom_(0, 0, 0, 0) = 1;
    this->bottom_(0, 0, 0, 1) = -1;
    this->bottom_(0, 0, 0, 2) = 2;

    this->bottom_(1, 0, 0, 0) = -3;
    this->bottom_(1, 0, 0, 1) = 1;
    this->bottom_(1, 0, 0, 2) = 2;

    auto& w = *this->layer_->mutable_param()[0];
    w[0] = 1;
    w[1] = 2;
    w[2] = -1;

    w[3] = -1;
    w[4] = 2;
    w[5] = -2;

    auto& b = *this->layer_->mutable_param()[1];
    b[0] = 5; b[1] = 10;

    this->top_gradient_[0] = 1;
    this->top_gradient_[1] = 2;
    this->top_gradient_[2] = 3;
    this->top_gradient_[3] = 4;

    this->layer_->bprop(
            {&this->bottom_},
            {&this->bottom_gradient_},
            {&this->top_},
            {&this->top_gradient_});

    auto &dw = *this->layer_->mutable_gradient()[0];
    auto &db = *this->layer_->mutable_gradient()[1];

    auto &dy = this->top_gradient_;

    auto &x = this->bottom_;
    auto &dx = this->bottom_gradient_;

    int stride = dw.w_;

    TypeParam expected;

    expected = dy[0]*x[0] + dy[2]*x[stride];
    EXPECT_EQ(dw[0], expected); // -8

    expected = dy[0]*x[1] + dy[2]*x[stride+1];
    EXPECT_EQ(dw[1], expected); // 2

    expected = dy[0]*x[2] + dy[2]*x[stride+2];
    EXPECT_EQ(dw[2], expected); // 8

    expected = dy[1]*x[0] + dy[3]*x[stride];
    EXPECT_EQ(dw[3], expected); // -10

    expected = dy[1]*x[1] + dy[3]*x[stride+1];
    EXPECT_EQ(dw[4], expected); // 2

    expected = dy[1]*x[2] + dy[3]*x[stride+2];
    EXPECT_EQ(dw[5], expected); // 12

    expected = dy[0] + dy[2];
    EXPECT_EQ(db[0], expected); // 4

    expected = dy[1] + dy[3];
    EXPECT_EQ(db[1], expected); // 6

    // now for the bottom
    expected = dy[0]*w[0] + dy[1]*w[3];
    EXPECT_EQ(dx[0], expected);   // -1

    expected = dy[0]*w[1] + dy[1]*w[4];
    EXPECT_EQ(dx[1], expected);   // 6

    expected = dy[0]*w[2] + dy[1]*w[5];
    EXPECT_EQ(dx[2], expected);   // -5

    expected = dy[2]*w[0] + dy[3]*w[3];
    EXPECT_EQ(dx[3], expected);   // -1

    expected = dy[2]*w[1] + dy[3]*w[4];
    EXPECT_EQ(dx[4], expected);   // 14

    expected = dy[2]*w[2] + dy[3]*w[5];
    EXPECT_EQ(dx[5], expected);   // -11

#if 0
    std::ostringstream ss;
    ss << "\nw:\n";
    auto &w2 = *this->layer_->param()[0];
    for (int i = 0; i < w2.total_; i++)
    {
        if (i && i%3 == 0) ss << "\n";
        ss << w2[i] << " ";
    }

    ss << "\nx:\n";
    for (int i = 0; i < this->bottom_.total_; i++)
    {
        if (i && i%3 == 0) ss << "\n";
        ss << this->bottom_[i] << " ";
    }

    ss << "\ntop:\n";
    for (int i = 0; i < this->top_.total_; i++)
    {
        ss << this->top_[i] << " ";
    }
    ss << "\ntop gradient:\n";
    for (int i = 0; i < this->top_gradient_.total_; i++)
    {
        ss << this->top_gradient_[i] << " ";
    }

    ss << "\nbottom gradient:\n";
    for (int i = 0; i < this->bottom_gradient_.total_; i++)
    {
        ss << this->bottom_gradient_[i] << " ";
    }

    LOG(INFO) << ss.str();

    for (int i = 0; i < dw.total_; i++)
    {
        LOG(INFO) << "dw[" << i << "]" << dw[i];
    }

    for (int i = 0; i < db.total_; i++)
    {
        LOG(INFO) << "db[" << i << "]" << db[i];
    }
#endif
}


}  // namespace cnn
