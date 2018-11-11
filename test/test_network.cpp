#include <glog/logging.h>
#include <gtest/gtest.h>

#define private public
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
    Network<TypeParam> network(this->filename_);
    network.reshape();
    for (const auto& p : network.data_)
    {
        LOG(INFO) << p.first << ": " << p.second->shape_info();
    }
}

} // namespace cnn
