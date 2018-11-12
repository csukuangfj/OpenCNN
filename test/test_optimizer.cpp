#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/optimizer.hpp"

namespace cnn
{

template<typename Dtype>
class OptimizerTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(OptimizerTest, MyTypes);

TYPED_TEST(OptimizerTest, init)
{
    std::string filename = "../proto/optimizer.prototxt";
    Optimizer<TypeParam> opt(filename);
}

}  // namespace cnn
