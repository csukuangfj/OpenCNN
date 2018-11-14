#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnn/jet.hpp"

namespace cnn
{

template<typename Dtype>
class JetTest : public ::testing::Test
{
};


using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(JetTest, MyTypes);

template<typename Dtype>
Dtype add_scalar1(const Dtype& x) {Dtype y; y = x + Dtype(1); return y;}

template<typename Dtype>
Dtype add_scalar2(const Dtype& x) {Dtype y; y = Dtype(1) + x; return y;}

template<typename Dtype>
Dtype add_scalar3(const Dtype& x) {Dtype y(x); y += Dtype(1); return y;}

template<typename Dtype>
Dtype sub_scalar1(const Dtype& x) {Dtype y; y = x - Dtype(1); return y;}

template<typename Dtype>
Dtype sub_scalar2(const Dtype& x) {Dtype y; y = Dtype(1) - x; return y;}

template<typename Dtype>
Dtype sub_scalar3(const Dtype& x) {Dtype y(x); y -= Dtype(1); return y;}


TYPED_TEST(JetTest, scalar_add)
{
    Jet<TypeParam, 1> f(10, 0);

    // x = x + 1
    f = add_scalar1(f);
    EXPECT_EQ(f.a_, 11);
    EXPECT_EQ(f.v_[0], 1);

    // x = 1 + x
    f = add_scalar2(f);
    EXPECT_EQ(f.a_, 12);
    EXPECT_EQ(f.v_[0], 1);

    // x += 1
    f = add_scalar2(f);
    EXPECT_EQ(f.a_, 13);
    EXPECT_EQ(f.v_[0], 1);
}

TYPED_TEST(JetTest, scalar_sub)
{
    Jet<TypeParam, 1> f(10, 0);

    // x = x - 1
    f = sub_scalar1(f);
    EXPECT_EQ(f.a_, 9);
    EXPECT_EQ(f.v_[0], 1);

    // x = 1 - x
    f = sub_scalar2(f);
    EXPECT_EQ(f.a_, -8);
    EXPECT_EQ(f.v_[0], -1);

    // x -= 1
    f = sub_scalar3(f);
    EXPECT_EQ(f.a_, -9);
    EXPECT_EQ(f.v_[0], -1);
}

TYPED_TEST(JetTest, negate)
{
    Jet<TypeParam, 1> f(10, 0);

    // x = -x
    f = -f;
    EXPECT_EQ(f.a_, -10);
    EXPECT_EQ(f.v_[0], -1);
}

}  // namespace cnn

