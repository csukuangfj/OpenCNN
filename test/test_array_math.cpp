#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/array_math.hpp"

namespace cnn
{
template<typename Dtype>
class ArrayMathTest : public ::testing::Test
{};


using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ArrayMathTest, MyTypes);

TYPED_TEST(ArrayMathTest, ax_sub_by)
{
    Array<TypeParam> arr1;
    Array<TypeParam> arr2;

    arr1.init(1, 2, 3, 4);
    arr2.init(24, 1, 1, 1);

    set_to<TypeParam>(&arr1, 1);
    set_to<TypeParam>(&arr2, 1);

    double diff = 1000;

    diff = ax_sub_by_squared<TypeParam>(arr1.total_, 1, &arr1(0, 0, 0, 0), 1, &arr2[0]);
    EXPECT_EQ(diff, 0);

    double sum = 0;

    sum = ax_sub_by_squared<TypeParam>(arr1.total_, -1, &arr1(0, 0, 0, 0), 1, &arr2[0]);
    EXPECT_EQ(sum, 4*arr1.total_);
}

}  // namespace cnn
