// Copyright 2019. All Rights Reserved.
// Author: csukuangfj@gmail.com (Fangjun Kuang)

#ifndef CNN_AUTODIFF_JET_H_
#define CNN_AUTODIFF_JET_H_

/*
 * autodiff.
 *
 *
 * we take the idea from ceres-solver for automatic differentiation.
 *
 * It is mainly used for verifying the back propagation step.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace cnn {

template <typename Dtype, int N>
class ArrayWithOp : public std::array<Dtype, N> {};

template <typename Dtype, int N>
ArrayWithOp<Dtype, N> operator+(const ArrayWithOp<Dtype, N>& a,
                                const ArrayWithOp<Dtype, N>& b) {
  ArrayWithOp<Dtype, N> c;
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <typename Dtype, int N>
ArrayWithOp<Dtype, N> operator-(const ArrayWithOp<Dtype, N>& a,
                                const ArrayWithOp<Dtype, N>& b) {
  ArrayWithOp<Dtype, N> c;
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <typename Dtype, int N>
ArrayWithOp<Dtype, N> operator-(const ArrayWithOp<Dtype, N>& a) {
  ArrayWithOp<Dtype, N> c;
  for (int i = 0; i < N; ++i) {
    c[i] = -a[i];
  }
  return c;
}

template <typename Dtype, int N>
ArrayWithOp<Dtype, N> operator*(const ArrayWithOp<Dtype, N>& a, double s) {
  ArrayWithOp<Dtype, N> c;
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] * s;
  }
  return c;
}

template <typename Dtype, int N>
ArrayWithOp<Dtype, N> operator*(double s, const ArrayWithOp<Dtype, N>& a) {
  ArrayWithOp<Dtype, N> c;
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] * s;
  }
  return c;
}

template <typename Dtype, int N>
ArrayWithOp<Dtype, N> operator/(const ArrayWithOp<Dtype, N>& a, double s) {
  ArrayWithOp<Dtype, N> c;
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] / s;
  }
  return c;
}

/**@brief Jet for autodiff (forward mode).
 *
 * This class is based on
 * https://github.com/kashif/ceres-solver/blob/master/include/ceres/jet.h
 *
 * We use the same name `Jet` as in ceres-solver.
 */
template <typename Dtype, int N>
class Jet {
 public:
  /** @brief The default constructor.
   *
   * Both the value and the graident are set to 0.
   */
  Jet() : val_() { grad_.fill(0); }

  /** @brief Conversion from a scalar
   *
   * We set the gradient to 0 since we assume the scalar is
   * a constant and the derivative with respect
   * to a constant is 0.
   *
   * @param val The value of the variable.
   *
   * @note we omit the **explicit** specifier
   * since it is expected.
   */
  Jet(const Dtype& val)  // NOLINT
      : val_(val) {
    grad_.fill(0);
  }

  /** @brief Construct with a value and its derivative.
   *
   * @param val the value of variable
   * @param i the position to set the derivative
   * @param derivative the derivative of the variable
   */
  Jet(const Dtype& val, int i, Dtype derivative = 1) : val_(val) {
    grad_.fill(0);
    grad_.at(i) = derivative;
  }

  /** @brief Set the value and its derivative.
   *
   * @param val the value of the variable
   * @param i the position to set the derivative
   * @param derviative the derivative of the variable
   */
  void set(const Dtype& val, int i, Dtype derivative = 1) {
    val_ = val;
    grad_.fill(0);
    grad_.at(i) = derivative;
  }

  /** @brief For debug purpose only.
   * @return a string representation of the jet.
   */
  std::string to_string() const {
    std::ostringstream ss;
    ss << "[" << val_ << ", (";
    std::string sep;
    for (int i = 0; i < N; ++i) {
      ss << sep << grad_[i];
      sep = ", ";
    }
    ss << ")]";
    ss << "\n";
    return ss.str();
  }

  Jet& operator+=(const Jet& f) { return *this = *this + f; }
  Jet& operator-=(const Jet& f) { return *this = *this - f; }
  Jet& operator*=(const Jet& f) { return *this = *this * f; }
  Jet& operator/=(const Jet& f) { return *this = *this / f; }

  Jet& operator+=(const Dtype& s) { return *this = *this + s; }
  Jet& operator-=(const Dtype& s) { return *this = *this - s; }
  Jet& operator*=(const Dtype& s) { return *this = *this * s; }
  Jet& operator/=(const Dtype& s) { return *this = *this / s; }

  /** @brief Get the value of the jet.
   *
   * @return the value of the jet.
   */
  operator Dtype() const { return val_; }

  Dtype val_;                   //!< value
  ArrayWithOp<Dtype, N> grad_;  //!< gradient
};

template <typename Dtype, int N>
std::ostream& operator<<(std::ostream& os, const Jet<Dtype, N>& f) {
  os << f.to_string();
  return os;
}

/** @brief Negate a jet.
 *
 * @param f the input jet
 * @return a jet with (-value, -gradient) of the input jet
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator-(const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;
  res.val_ = -f.val_;
  res.grad_ = -f.grad_;
  return res;
}

//----------------------------------------
//  scalars
//  +, -, *, /
//----------------------------------------

/** @brief Jet + scalar
 *
 * @param f the input jet
 * @param s the scalar
 *
 * @note the gradient is not changed and only the value is
 * increased by `s`.
 *
 * @return a jet with (s + value, gradient)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator+(const Jet<Dtype, N>& f, Dtype s) {
  Jet<Dtype, N> res(f);
  res.val_ += s;
  return res;
}

/** @brief scalar + jet
 *
 * @param s the scalar
 * @param f the in jet
 *
 * @note the gradient is not changed and only the value
 * is increased by `s`.
 *
 * @return a jet with (s + value, gradient)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator+(Dtype s, const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res(f);
  res.val_ += s;
  return res;
}

/** @brief jet - scalar
 *
 * @param f the input jet
 * @param s the scalar
 *
 * @note the gradient is not changed
 *
 * @return a jet with (val - s, gradient)
 *
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator-(const Jet<Dtype, N>& f, Dtype s) {
  Jet<Dtype, N> res(f);
  res.val_ -= s;
  return res;
}

/** @brief scalar - jet
 *
 * @param s the scalar
 * @param f the input jet
 *
 * @note the gradient is negated
 *
 * @return a jet with (s - value, -gradient)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator-(Dtype s, const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;
  res.val_ = s - f.val_;
  res.grad_ = -f.grad_;
  return res;
}

/** @brief jet * scalar
 *
 * @param f the input jet
 * @param s the scalar
 *
 * @note the gradient is scaled by s
 *
 * @return a jet with (s * value, s * gradient)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator*(const Jet<Dtype, N>& f, Dtype s) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ * s;
  res.grad_ = f.grad_ * s;
  return res;
}

/** @brief scalar * jet
 *
 * @param s the scalar
 * @param f the input jet
 *
 * @note the gradient is scaled by s
 *
 * @return a jet with (s * value, s * gradient)
 *
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator*(Dtype s, const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ * s;
  res.grad_ = f.grad_ * s;
  return res;
}

/** @brief jet / s
 *
 * @param f the input jet
 * @param s the scalar
 *
 * @note the gradient is scaled by (1/s)
 *
 * @return a jet with (value/s, gradient/s)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator/(const Jet<Dtype, N>& f, Dtype s) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ / s;
  res.grad_ = f.grad_ / s;
  return res;
}

/** @brief \f$\frac{s}{jet}\f$
 * \f[
 * \frac{s}{x + \epsilon\cdot g} = \frac{s(x - \epsilon\cdot g)}{(x +
 * \epsilon\cdot g)(x - \epsilon\cdot g)} = \frac{s \cdot x - \epsilon\cdot s
 * g}{x^2 - \epsilon^2\cdot g^2}
 * \f]
 *
 * Since \f$\epsilon^2 == 0\f$, we have
 *
 * \f[
 * \frac{s}{x + \epsilon \cdot g} = \frac{s \cdot x - \epsilon \cdot sg}{x^2}
 * = \frac{s}{x} - \epsilon \frac{sg}{x^2}
 * \f]
 *
 * Therefore, we should return a jet with \f$(\frac{s}{\mathrm{value}},
 * \frac{s}{\mathrm{value}^2}\cdot \mathrm{gradient})\f$
 *
 * @return a jet with \f$(\frac{s}{\mathrm{value}},
 * \frac{s}{\mathrm{value}^2}\cdot \mathrm{gradient})\f$
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator/(Dtype s, const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;
  res.val_ = s / f.val_;
  res.grad_ = -s * f.grad_ / (f.val_ * f.val_);
  return res;
}

/** @brief jet + jet
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @return (f.val + g.val, f.grad + g.grad)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator+(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ + g.val_;
  res.grad_ = f.grad_ + g.grad_;
  return res;
}

/** @brief jet - jet
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @return (f.val - g.val, f.grad - g.grad)
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator-(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ - g.val_;
  res.grad_ = f.grad_ - g.grad_;
  return res;
}
/** @bref jet * jet
 *
 * @param f the first jet
 * @param g the second jet
 *
 * \f[
 * (x + \epsilon f_x)\cdot(y + \epsilon f_y) =
 * xy + \epsilon (x\cdot f_y + y\cdot f_x) + \epsilon^2 (f_x \cdot f_y)
 * \f]
 *
 * Since \f$\epsilon^2 == 0\f$, we have
 *
 * \f[
 * (x + \epsilon f_x)\cdot(y + \epsilon f_y) =
 * xy + \epsilon (x\cdot f_y + y\cdot f_x)
 * \f]
 *
 * So we should return a jet \f$(\mathrm{f.val} \times \mathrm{g.val},
 * x\times\mathrm{g.grad} + y\times\mathrm{f.grad})\f$
 *
 * @return a jet \f$(\mathrm{f.val} \times\mathrm{g.val},
 * \mathrm{f.val}\times\mathrm{g.grad} + \mathrm{g.val}\times\mathrm{f.grad})\f$
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator*(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ * g.val_;
  res.grad_ = f.val_ * g.grad_ + g.val_ * f.grad_;
  return res;
}

/** @brief \frac{jet}{jet}
 *
 * @param f the first jet (numerator)
 * @param g the second jet (denominator)
 *
 * \f[
 * \frac{x + \epsilon \cdot f_x}{y + \epsilon \cdot f_y}
 * =\frac{(x+\epsilon \cdot f_x)(y - \epsilon \cdot f_y)}{(y+\epsilon\cdot
 * f_y)(y-\epsilon\cdot f_y)}
 * = \frac{x\cdot y + \epsilon(y\cdot f_x - x\cdot f_y) - \epsilon^2 f_x
 * f_y}{y^2 - \epsilon^2 f_y^2}
 * \f]
 *
 * Since \f$\epsilon^2 == 0\f$, we have
 * \f[
 * \frac{x + \epsilon \cdot f_x}{y + \epsilon \cdot f_y}
 * = \frac{x\cdot y + \epsilon(y\cdot f_x - x\cdot f_y)}{y^2}
 * = \frac{x}{y} + \epsilon (\frac{f_x}{y} - \frac{x \cdot f_y}{y^2})
 * \f]
 *
 * @return a jet \f$(\frac{\mathrm{f.val}}{\mathrm{g.val}},
 * \frac{\mathrm{f.grad}}{\mathrm{g.val}} -
 * \frac{\mathrm{f.val}\times\mathrm{g.grad}}{\mathrm{g.val}^2})\f$
 */
template <typename Dtype, int N>
Jet<Dtype, N> operator/(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  Jet<Dtype, N> res;
  res.val_ = f.val_ / g.val_;
  res.grad_ = f.grad_ / g.val_ - (f.val_ * g.grad_) / (g.val_ * g.val_);
  return res;
}

/** @brief compare two jets: jet1 == jet2
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning We only consider the value; the gradient of the jet
 * is not considered.
 *
 * @return true if the values of the two jet are equal; false otherwise
 */
template <typename Dtype, int N>
bool operator==(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return f.val_ == g.val_;
}

/** @brief Compare two jets: jet1 != jet2
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning We only consider the value; the gradient of the jet
 * is not considered.
 *
 * @return true if the values of the two jet are **NOT** equal; false otherwise
 */
template <typename Dtype, int N>
bool operator!=(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return f.val_ != g.val_;
}

/** @brief Compare two jets: jet1 < jet2
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning We only consider the value; the gradient of the jet
 * is not considered.
 *
 * @return true if f.val < g.val; false otherwise
 */
template <typename Dtype, int N>
bool operator<(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return f.val_ < g.val_;
}

/** @brief Compare two jets: jet1 <= jet2
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning We only consider the value; the gradient of the jet
 * is not considered.
 *
 * @return true if f.val <= g.val; false otherwise
 */
template <typename Dtype, int N>
bool operator<=(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return f.val_ <= g.val_;
}

/** @brief Compare two jets: jet1 > jet2
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning We only consider the value; the gradient of the jet
 * is not considered.
 *
 * @return true if f.val > g.val; false otherwise
 */
template <typename Dtype, int N>
bool operator>(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return f.val_ > g.val_;
}

/** @brief Compare two jets: jet1 >= jet2
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning We only consider the value; the gradient of the jet
 * is not considered.
 *
 * @return true if f.val >= g.val; false otherwise
 */
template <typename Dtype, int N>
bool operator>=(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return f.val_ >= g.val_;
}

//--------------------
//  math functions
//      std::exp
//      std::log
//      std::max
//      std::sqrt
//--------------------
using std::max;

/** @brief max(jet1, jet2)
 *
 * @param f the first jet
 * @param g the second jet
 *
 * @warning It returns by value!
 *
 * @return g if f.val < g.val; f otherwise
 */
template <typename Dtype, int N>
Jet<Dtype, N> max(const Jet<Dtype, N>& f, const Jet<Dtype, N>& g) {
  return (f < g) ? g : f;
}

using std::exp;

/** @brief exp(jet)
 *
 * Assume \f$f = x + \epsilon \cdot f_x\f$, then the derivative
 * of \f$\mathrm{e}^f\f$ with respect to x is \f$\mathrm{e}^f \cdot f_x\f$.
 *
 * Thus, we should return a jet (exp(value), exp(value)*grad)
 *
 * @param f the input jet
 *
 * @return a jet with (exp(value), exp(value)*grad)
 */
template <typename Dtype, int N>
Jet<Dtype, N> exp(const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;

  auto s = exp(f.val_);
  res.val_ = s;
  res.grad_ = s * f.grad_;

  return res;
}

using std::log;

/** @brief log(jet)
 *
 * Assume \f$f = x + \epsilon \cdot f_x\f$, then the derivative
 * of \f$\mathrm{log}(f)\f$ with respect to x is \f$\frac{1}{f} \cdot f_x\f$.
 *
 * Thus, we should return a jet (log(value),
 * \f$\frac{1}{\mathrm{value}}\f$*grad)
 *
 * @return a jet (log(value),
 * \f$\frac{1}{\mathrm{value}}\f$*grad)
 */
template <typename Dtype, int N>
Jet<Dtype, N> log(const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;

  res.val_ = log(f.val_);
  res.grad_ = f.grad_ / f.val_;

  return res;
}

using std::sqrt;

/** @brief sqrt(jet)
 *
 * Assume \f$f = x + \epsilon \cdot f_x\f$, then the derivative
 * of \f$\sqrt{f}\f$ with respect to x is \f$\frac{1}{2\sqrt{f}} \cdot f_x\f$.
 *
 * Thus, we should return a jet \f$(\sqrt{\mathrm{value}},
 * \frac{1}{\sqrt{\mathrm{value}}}\f$*grad)
 *
 * @return a jet \f$(\sqrt{\mathrm{value}},
 * \frac{1}{\sqrt{\mathrm{value}}}\f$*grad)
 */
template <typename Dtype, int N>
Jet<Dtype, N> sqrt(const Jet<Dtype, N>& f) {
  Jet<Dtype, N> res;

  res.val_ = sqrt(f.val_);
  res.grad_ = f.grad_ / (Dtype(2) * res.val_);

  return res;
}

}  // namespace cnn
#endif  // CNN_AUTODIFF_JET_H_
