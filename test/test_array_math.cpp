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

TYPED_TEST(ArrayMathTest, scale_arr)
{
    Array<TypeParam> d;
    d.init(1, 2, 1, 1);
    d[0] = 2;
    d[1] = 3;
    scale_arr<TypeParam>(2, d, &d);
    EXPECT_EQ(d[0], 4);
    EXPECT_EQ(d[1], 6);

    scale_arr<TypeParam>(0.5, d, &d);
    EXPECT_EQ(d[0], 2);
    EXPECT_EQ(d[1], 3);
}

TYPED_TEST(ArrayMathTest, sum_arr)
{
    Array<TypeParam> d;
    d.init(1, 2, 1, 1);
    d[0] = -1;
    d[1] = 3;
    auto r = sum_arr<TypeParam>(d);
    EXPECT_EQ(r, 2);
}

TYPED_TEST(ArrayMathTest, sum_arr2)
{
    Array<TypeParam> d;
    d.init(1, 2, 1, 1);
    d[0] = -2;
    d[1] = 3;
    auto r = sum_arr<TypeParam>(d.total_, &d[0]);
    EXPECT_EQ(r, 1);
}

TYPED_TEST(ArrayMathTest, sub_scalar)
{
    Array<TypeParam> d;
    d.init(1, 2, 1, 1);
    d[0] = 10;
    d[1] = 5;

    sub_scalar<TypeParam>(2, d, &d);
    EXPECT_EQ(d[0], 8);
    EXPECT_EQ(d[1], 3);
}

TYPED_TEST(ArrayMathTest, sub_scalar2)
{
    Array<TypeParam> d;
    d.init(1, 2, 1, 1);
    d[0] = 10;
    d[1] = 5;

    sub_scalar<TypeParam>(d.total_, 3, &d[0], &d[0]);
    EXPECT_EQ(d[0], 7);
    EXPECT_EQ(d[1], 2);
}

}  // namespace cnn
