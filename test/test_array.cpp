#include <gtest/gtest.h>
#include <glog/logging.h>

#include "cnn/array.hpp"

namespace cnn
{

template<typename Dtype>
class ArrayTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ArrayTest, MyTypes);

template<typename Dtype>
bool isAllZeros(const Array<Dtype> &arr)
{
    for (int i = 0; i < arr.total_; i++)
    {
        if (arr.d_[i]) return false;
    }

    return true;
}

// in side the function, use
// "TypeParam" to refer to the data type
TYPED_TEST(ArrayTest, default_constructor)
{
    Array<TypeParam> arr;
    EXPECT_EQ(arr.d_, nullptr);
    EXPECT_EQ(arr.n_, 0);
    EXPECT_EQ(arr.c_, 0);
    EXPECT_EQ(arr.h_, 0);
    EXPECT_EQ(arr.w_, 0);
    EXPECT_EQ(arr.total_, 0);
}

TYPED_TEST(ArrayTest, init)
{
    Array<TypeParam> arr;

    arr.init(1, 1, 1, 2);
    EXPECT_NE(arr.d_, nullptr);
    EXPECT_EQ(arr.n_, 1);
    EXPECT_EQ(arr.c_, 1);
    EXPECT_EQ(arr.h_, 1);
    EXPECT_EQ(arr.w_, 2);
    EXPECT_EQ(arr.total_, 2);
    EXPECT_EQ(isAllZeros(arr), true);

    TypeParam *d = arr.d_;
    arr.init(1, 1, 2, 1);
    EXPECT_EQ(arr.d_, d);   // no memory is re-allocated
    EXPECT_EQ(arr.n_, 1);
    EXPECT_EQ(arr.c_, 1);
    EXPECT_EQ(arr.h_, 2);
    EXPECT_EQ(arr.w_, 1);
    EXPECT_EQ(arr.total_, 2);
    EXPECT_EQ(isAllZeros(arr), true);

    arr.init(2, 3, 4, 5);
    EXPECT_NE(arr.d_, d);   // memory is re-allocated
    EXPECT_EQ(arr.n_, 2);
    EXPECT_EQ(arr.c_, 3);
    EXPECT_EQ(arr.h_, 4);
    EXPECT_EQ(arr.w_, 5);
    EXPECT_EQ(arr.total_, 120);
    EXPECT_EQ(isAllZeros(arr), true);

    d = arr.d_;
    arr.init(3, 2, 5, 4);
    EXPECT_EQ(arr.d_, d);   // no memory is re-allocated
    EXPECT_EQ(arr.n_, 3);
    EXPECT_EQ(arr.c_, 2);
    EXPECT_EQ(arr.h_, 5);
    EXPECT_EQ(arr.w_, 4);
    EXPECT_EQ(arr.total_, 120);
    EXPECT_EQ(isAllZeros(arr), true);
}

TYPED_TEST(ArrayTest, at)
{
    Array<TypeParam> arr;
    arr.init(2, 3, 4, 5);
    for (int i = 0; i < arr.total_; i++) arr.d_[i] = i;

    EXPECT_EQ(arr.at(1, 2, 3, 4), 1*3*4*5 + 2*4*5 + 3*5 + 4);
    EXPECT_EQ(arr.at(0, 0, 0, 0), 0);
    EXPECT_EQ(arr.at(0, 0, 0, 4), 4);
    ASSERT_DEATH(arr.at(2, 0, 0, 0), "Check failed: n < n_");

    arr.init(1, 2, 3, 4);
    arr.at(0, 0, 0, 3) = 3;
    EXPECT_EQ(arr.d_[3], 3);
}

TYPED_TEST(ArrayTest, operator_paren)
{
    Array<TypeParam> arr;
    arr.init(2, 3, 4, 5);
    for (int i = 0; i < arr.total_; i++) arr.d_[i] = i;

    EXPECT_EQ(arr(1, 2, 3, 4), 1*3*4*5 + 2*4*5 + 3*5 + 4);
    EXPECT_EQ(arr(0, 0, 0, 0), 0);
    EXPECT_EQ(arr(0, 0, 0, 4), 4);

    arr.init(1, 2, 3, 4);
    arr(0, 0, 0, 3) = 3;
    EXPECT_EQ(arr.d_[3], 3);
}

TYPED_TEST(ArrayTest, operator_bracket)
{
    Array<TypeParam> arr;
    arr.init(2, 3, 4, 5);
    for (int i = 0; i < arr.total_; i++) arr.d_[i] = i;

    EXPECT_EQ(arr[10], 10);
    EXPECT_EQ(arr[100], 100);

    arr.d_[100] = 0;
    EXPECT_EQ(arr[100], 0);
}

}  // namespace cnn
