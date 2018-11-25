#include <gtest/gtest.h>

#include "cnn/rng.hpp"

namespace cnn
{

template<typename Dtype>
class RngTest : public ::testing::Test
{};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(RngTest, MyTypes);

TYPED_TEST(RngTest, gaussian)
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

TYPED_TEST(RngTest, uniform)
{
    set_seed(100);
    Array<TypeParam> arr;
    arr.init(100, 100, 10, 5);

    TypeParam low = -100;
    TypeParam high = 200;
    uniform<TypeParam>(&arr, low, high);
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

    TypeParam expected_mean = (low + high) / 2;
    EXPECT_NEAR(mean, expected_mean, expected_mean*1e-1);

    TypeParam expected_var = (high - low)*(high - low) / 12;
    EXPECT_NEAR(var, expected_var, expected_var*1e-1);
}

TYPED_TEST(RngTest, bernoulli)
{
    set_seed(1989);
    Array<bool> arr;
    arr.init(100, 100, 10, 5);

    double p = 0.8;

    bernoulli<bool>(&arr, p);

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

    TypeParam expected_mean = p;
    EXPECT_NEAR(mean, expected_mean, expected_mean*1e-2);

    TypeParam expected_var = p * (1 - p);
    EXPECT_NEAR(var, expected_var, expected_var*1e-2);
}

}  // namespace cnn

