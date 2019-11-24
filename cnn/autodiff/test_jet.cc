// Copyright 2019. All Rights Reserved.
// Author: csukuangfj@gmail.com (Fangjun Kuang)

#include "cnn/autodiff/jet.h"

#include "gtest/gtest.h"

namespace cnn {

template <typename Dtype>
class JetTest : public ::testing::Test {};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(JetTest, MyTypes);

template <typename Dtype>
Dtype scalar_add_jet(typename Dtype::type s, const Dtype& x) {
  return s + x;
}

template <typename Dtype>
Dtype jet_add_scalar(const Dtype& x, typename Dtype::type s) {
  return x + s;
}

template <typename Dtype>
Dtype inc_by_one(const Dtype& x) {
  Dtype y = x;
  y += typename Dtype::type(1);
  return y;
}

template <typename Dtype>
Dtype scalar_sub_jet(typename Dtype::type s, const Dtype& x) {
  return s - x;
}

template <typename Dtype>
Dtype jet_sub_scalar(const Dtype& x, typename Dtype::type s) {
  return x - s;
}

template <typename Dtype>
Dtype dec_by_one(const Dtype& x) {
  Dtype y = x;
  y -= typename Dtype::type(1);
  return y;
}

template <typename Dtype>
Dtype scalar_mul_jet(typename Dtype::type s, const Dtype& x) {
  return s * x;
}

template <typename Dtype>
Dtype jet_mul_scalar(const Dtype& x, typename Dtype::type s) {
  return x * s;
}

template <typename Dtype>
Dtype mul_by_ten(const Dtype& x) {
  Dtype y = x;
  y *= typename Dtype::type(10);
  return y;
}

template <typename Dtype>
Dtype scalar_div_jet(typename Dtype::type s, const Dtype& x) {
  return s / x;
}

template <typename Dtype>
Dtype jet_div_scalar(const Dtype& x, typename Dtype::type s) {
  return x / s;
}

template <typename Dtype>
Dtype div_by_two(const Dtype& x) {
  Dtype y = x;
  y /= typename Dtype::type(2);
  return y;
}

template <typename Dtype>
Dtype jet_mul_div_add(const Dtype& x, const Dtype& y) {
  auto z = x * y / (x * x + y);
  return z;
  // from https://www.whitman.edu/mathematics/calculus_online/section14.03.html
  /*
   * dz/dx = (y*y - x*x*y)/((x*x+y)*(x*x+y))
   * dz/dy = x*x*x/((x*x+y)*(x*x+y))
   */
}

template <typename Dtype>
Dtype jet_sub_div(const Dtype& x, const Dtype& y) {
  auto z = (x - y) / (x + y);
  return z;
  // http://math.gmu.edu/~memelian/teaching/Fall08/partDerivExamples.pdf
  // 1 (e)
  /*
   * dz/dx = 2y/((x+y)*(x+y))
   * dz/dy = -2x/((x+y)*(x+y))
   */
}

TYPED_TEST(JetTest, add_scalar) {
  Jet<TypeParam> f(Dim(1), 10, 0);

  // x = x + 1.25
  f = jet_add_scalar(f, 1.25);
  EXPECT_EQ(f.val_, 11.25);
  EXPECT_EQ(f.grad_[0], 1);

  // x = 1 + x
  f = scalar_add_jet(1, f);
  EXPECT_EQ(f.val_, 12.25);
  EXPECT_EQ(f.grad_[0], 1);

  // x += 1
  f = inc_by_one(f);
  EXPECT_EQ(f.val_, 13.25);
  EXPECT_EQ(f.grad_[0], 1);
}

TYPED_TEST(JetTest, sub_scalar) {
  Jet<TypeParam> f(Dim(1), 10, 0);

  // x = x - 1.25
  f = jet_sub_scalar(f, 1.25);
  EXPECT_EQ(f.val_, 8.75);
  EXPECT_EQ(f.grad_[0], 1);

  // x = 1 - x
  f = scalar_sub_jet(1, f);
  EXPECT_EQ(f.val_, -7.75);
  EXPECT_EQ(f.grad_[0], -1);

  // x -= 1
  f = dec_by_one(f);
  EXPECT_EQ(f.val_, -8.75);
  EXPECT_EQ(f.grad_[0], -1);
}

TYPED_TEST(JetTest, negate) {
  Jet<TypeParam> f(Dim(1), 10, 0);

  // x = -x
  f = -f;
  EXPECT_EQ(f.val_, -10);
  EXPECT_EQ(f.grad_[0], -1);
}

TYPED_TEST(JetTest, mul_scalar) {
  Jet<TypeParam> f(Dim(1), 10, 0);

  // x = x * 2
  f = jet_mul_scalar(f, 2);
  EXPECT_EQ(f.val_, 20);
  EXPECT_EQ(f.grad_[0], 2);

  // x = 5 * x
  f = scalar_mul_jet(5, f);
  EXPECT_EQ(f.val_, 100);
  EXPECT_EQ(f.grad_[0], 10);

  // x *= 10
  f = mul_by_ten(f);
  EXPECT_EQ(f.val_, 1000);
  EXPECT_EQ(f.grad_[0], 100);
}

TYPED_TEST(JetTest, scalar_div) {
  Jet<TypeParam> f(Dim(1), 10, 0);

  // x = x / 2
  f = jet_div_scalar(f, 2);
  EXPECT_EQ(f.val_, 5);
  EXPECT_EQ(f.grad_[0], 0.5);

  // x = 2 / x
  f = scalar_div_jet(2, f);
  EXPECT_NEAR(f.val_, 0.4, 1e-7);
  EXPECT_EQ(f.grad_[0], -TypeParam(2) / 25 * 0.5);

  // x /= 2
  f = div_by_two(f);
  EXPECT_NEAR(f.val_, 0.2, 1e-7);
  EXPECT_EQ(f.grad_[0], -TypeParam(2) / 25 * 0.5 * 0.5);
}

TYPED_TEST(JetTest, jet_mul_div_add_test) {
  TypeParam x = 1;
  TypeParam y = 2;
  Jet<TypeParam> a(Dim(2), x, 0);
  Jet<TypeParam> b(Dim(2), y, 1);
  auto z = jet_mul_div_add(a, b);

  EXPECT_NEAR(z.val_, jet_mul_div_add(x, y), 1e-6);
  TypeParam dz_dx;
  TypeParam dz_dy;

  dz_dx = (y * y - x * x * y) / ((x * x + y) * (x * x + y));
  dz_dy = x * x * x / ((x * x + y) * (x * x + y));
  EXPECT_NEAR(z.grad_[0], dz_dx, 1e-7);
  EXPECT_NEAR(z.grad_[1], dz_dy, 1e-7);
}

TYPED_TEST(JetTest, jet_sub_div) {
  TypeParam x = 1;
  TypeParam y = 2;
  Jet<TypeParam> a(Dim(2), x, 0);
  Jet<TypeParam> b(Dim(2), y, 1);
  auto z = jet_sub_div(a, b);

  EXPECT_NEAR(z.val_, jet_sub_div(x, y), 1e-7);
  TypeParam dz_dx;
  TypeParam dz_dy;

  dz_dx = 2 * y / ((x + y) * (x + y));
  dz_dy = -2 * x / ((x + y) * (x + y));
  EXPECT_NEAR(z.grad_[0], dz_dx, 1e-7);
  EXPECT_NEAR(z.grad_[1], dz_dy, 1e-7);
}

TYPED_TEST(JetTest, jet_exp_log) {
  using Type = Jet<TypeParam>;
  Type x(Dim(1), exp(TypeParam(2)), 0, 10);

  {
    auto f = exp(x);
    auto g = log(f);
    EXPECT_NEAR(g.val_, x.val_, 1e-6);
    EXPECT_NEAR(g.grad_[0], x.grad_[0], 1e-6);
  }

  {
    auto f = log(x);
    auto g = exp(f);
    EXPECT_NEAR(g.val_, x.val_, 1e-6);
    EXPECT_NEAR(g.grad_[0], x.grad_[0], 1e-6);
  }
}

TYPED_TEST(JetTest, jet_sqrt) {
  using Type = Jet<TypeParam>;
  Type x(Dim(1), exp(TypeParam(2)), 0, 10);

  {
    auto f = sqrt(x);
    auto g = f * f;
    EXPECT_NEAR(g.val_, x.val_, 1e-6);
    EXPECT_NEAR(g.grad_[0], x.grad_[0], 1e-6);
  }

  {
    auto f = x * x;
    auto g = sqrt(f);
    EXPECT_NEAR(g.val_, x.val_, 1e-6);
    EXPECT_NEAR(g.grad_[0], x.grad_[0], 1e-6);
  }
}

}  // namespace cnn
