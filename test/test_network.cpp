#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stdlib.h>     // for srand()

#define private public

#include "cnn/array_math.hpp"
#include "cnn/network.hpp"
#include "proto/cnn.pb.h"

namespace cnn
{

template<typename Dtype>
class NetworkTest : public ::testing::Test
{
    void SetUp() override
    {
        filename_ = "../proto/model.prototxt";
    }
protected:
    std::string filename_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(NetworkTest, MyTypes);

TYPED_TEST(NetworkTest, constructor)
{
    srand(1989);
    Network<TypeParam> network(this->filename_);
    network.reshape();
    for (const auto& p : network.data_)
    {
        LOG(INFO) << p.first << ": " << p.second->shape_info();
    }
}

TYPED_TEST(NetworkTest, fprop)
{
    srand(1989);
    Network<TypeParam> network(this->filename_);
    network.reshape();

    auto input = network.get_data_output(0);
    input[0]->operator()(0, 0, 0, 0) = 1;
    input[0]->operator()(0, 0, 0, 1) = 2;
    input[0]->operator()(0, 0, 0, 2) = 3;

    input[0]->operator()(1, 0, 0, 0) = 4;
    input[0]->operator()(1, 0, 0, 1) = 5;
    input[0]->operator()(1, 0, 0, 2) = 6;

    input[1]->d_[0] = 100;
    input[1]->d_[1] = 200;

    auto param = network.layer(1)->param();
    // test only one output
    param[0]->d_[0] = 10;
    param[0]->d_[1] = 20;
    param[0]->d_[2] = 30;

    param[1]->d_[0] = 2;

    network.fprop();

    auto output = network.get_data_output(1)[0];
    // 1*10+2*20+3*30+2 = 10+40+90+2=142
    EXPECT_EQ(output->d_[0], 142);

    // 40+100+180+2=322
    EXPECT_EQ(output->d_[1], 322);

    auto loss = network.get_data_output(2)[0]->d_[0];
    TypeParam expected_loss;
    expected_loss = (142-100)*(142-100) + (322-200)*(322-200);
    EXPECT_EQ(loss, expected_loss);
}

} // namespace cnn
