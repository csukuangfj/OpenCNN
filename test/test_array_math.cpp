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

TYPED_TEST(ArrayMathTest, ax_sub_by_squared)
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

TYPED_TEST(ArrayMathTest, ax_dot_by)
{
    Array<TypeParam> arr1;
    Array<TypeParam> arr2;

    arr1.init(1, 1, 2, 3);
    arr2.init(1, 1, 2, 3);

    set_to<TypeParam>(&arr1, 1);
    set_to<TypeParam>(&arr2, 1);

    TypeParam sum = 0;
    sum = ax_dot_by<TypeParam>(arr1.total_, 1, arr1.d_, 1, arr2.d_);
    EXPECT_EQ(sum, arr1.total_);

    sum = 0;
    set_to<TypeParam>(&arr2, 2);
    sum = ax_dot_by<TypeParam>(arr1.total_, -1, arr1.d_, 2, arr2.d_);
    EXPECT_EQ(sum, -arr1.total_*4);
}

TYPED_TEST(ArrayMathTest, gaussian)
{
    set_seed(200);
    Array<TypeParam> arr;
    arr.init(100, 100, 10, 5);
    gaussian<TypeParam>(&arr, 1, 5);
    TypeParam mean = 0;
    TypeParam var = 0;
    for (int i = 0; i < arr.total_; i++)
    {
        mean += arr[i];
    }
    mean /= arr.total_;

    for (int i = 0; i < arr.total_; i++)
    {
        TypeParam diff = arr[i] - mean;
        var += diff*diff;
    }
    var /= arr.total_;
    EXPECT_NEAR(mean, 1, 0.01);
    EXPECT_NEAR(5/std::sqrt(var), 1, 0.01);
}

}  // namespace cnn
