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

TYPED_TEST(ArrayTest, operators)
{
    Array<TypeParam> arr;
    arr.init(2, 3, 1, 5);

    int i = 0;
    for (int n = 0; n < 2; n++)
    for (int c = 0; c < 3; c++)
    for (int h = 0; h < 1; h++)
    for (int w = 0; w < 5; w++)
    {
        arr(n, c, h, w) = i;
        i++;
    }

    for (int k = 0; k < arr.total_; k++)
    {
        EXPECT_EQ(arr[k], k);
    }

    arr.init(5, 6, 7, 8);
    for (int k = 0; k < arr.total_; k++)
    {
        arr[k] = k;
    }

    i = 0;
    for (int n = 0; n < 5; n++)
    for (int c = 0; c < 6; c++)
    for (int h = 0; h < 7; h++)
    for (int w = 0; w < 8; w++)
    {
        EXPECT_EQ(arr(n, c, h, w), i);
        i++;
    }
}

TYPED_TEST(ArrayTest, proto)
{
    Array<TypeParam> arr;

    arr.init(2, 3, 4, 5);
    for (int i = 0; i < arr.total_; i++) arr.d_[i] = i;

    ArrayProto proto;
    arr.to_proto(&proto);

    EXPECT_EQ(arr.n_, proto.n());
    EXPECT_EQ(arr.c_, proto.c());
    EXPECT_EQ(arr.h_, proto.h());
    EXPECT_EQ(arr.w_, proto.w());
    for (int i = 0; i < arr.total_; i++)
    {
        EXPECT_EQ(arr[i], proto.d(i));
    }

    Array<TypeParam> arr2;
    arr2.from_proto(proto);

    EXPECT_EQ(arr.n_, arr2.n_);
    EXPECT_EQ(arr.c_, arr2.c_);
    EXPECT_EQ(arr.h_, arr2.h_);
    EXPECT_EQ(arr.w_, arr2.w_);
    EXPECT_EQ(arr.total_, arr2.total_);

    for (int i = 0; i < arr.total_; i++)
    {
        EXPECT_EQ(arr[i], arr2[i]);
    }
}

TYPED_TEST(ArrayTest, move_constructor)
{
    Array<TypeParam> a;

    a.init(2, 3, 4, 5);
    for (int i = 0; i < a.total_; i++)
    {
        a[i] = i;
    }

    Array<TypeParam> b = std::move(a);
    EXPECT_EQ(a.n_, 0);
    EXPECT_EQ(a.c_, 0);
    EXPECT_EQ(a.h_, 0);
    EXPECT_EQ(a.w_, 0);
    EXPECT_EQ(a.total_, 0);
    EXPECT_EQ(a.d_, nullptr);

    EXPECT_EQ(b.n_, 2);
    EXPECT_EQ(b.c_, 3);
    EXPECT_EQ(b.h_, 4);
    EXPECT_EQ(b.w_, 5);
    EXPECT_EQ(b.total_, 2*3*4*5);
    for (int i = 0; i < b.total_; i++)
    {
        EXPECT_EQ(b[i], i);
    }
}

TYPED_TEST(ArrayTest, move_assignment)
{
    Array<TypeParam> a;

    a.init(2, 3, 4, 5);
    for (int i = 0; i < a.total_; i++)
    {
        a[i] = i;
    }

    Array<TypeParam> b;
    b = std::move(a);

    EXPECT_EQ(a.n_, 0);
    EXPECT_EQ(a.c_, 0);
    EXPECT_EQ(a.h_, 0);
    EXPECT_EQ(a.w_, 0);
    EXPECT_EQ(a.total_, 0);
    EXPECT_EQ(a.d_, nullptr);

    EXPECT_EQ(b.n_, 2);
    EXPECT_EQ(b.c_, 3);
    EXPECT_EQ(b.h_, 4);
    EXPECT_EQ(b.w_, 5);
    EXPECT_EQ(b.total_, 2*3*4*5);
    for (int i = 0; i < b.total_; i++)
    {
        EXPECT_EQ(b[i], i);
    }
}

}  // namespace cnn

