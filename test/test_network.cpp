#include <gtest/gtest.h>
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
}

} // namespace cnn
