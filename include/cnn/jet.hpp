#pragma once

/*
 * autodiff.
 *
 * refer to
 * https://github.com/kashif/ceres-solver/blob/master/include/ceres/jet.h
 *
 * we take the idea from ceres-solver
 * for automatic differentiation.
 *
 * It is mainly used for verifying the back propagation step.
 */

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace cnn
{
template<typename Dtype, int N>
class ArrayWithOp : public std::array<Dtype, N>
{
};

template<typename Dtype, int N>
ArrayWithOp<Dtype, N> operator+(
        const ArrayWithOp<Dtype, N>& a,
        const ArrayWithOp<Dtype, N>& b)
{
    ArrayWithOp<Dtype, N> c;
    for (int i = 0; i < N; i++) c[i] = a[i] + b[i];
    return c;
}

template<typename Dtype, int N>
ArrayWithOp<Dtype, N> operator-(
        const ArrayWithOp<Dtype, N>& a,
        const ArrayWithOp<Dtype, N>& b)
{
    ArrayWithOp<Dtype, N> c;
    for (int i = 0; i < N; i++) c[i] = a[i] - b[i];
    return c;
}

template<typename Dtype, int N>
ArrayWithOp<Dtype, N> operator-(const ArrayWithOp<Dtype, N>& a)
{
    ArrayWithOp<Dtype, N> c;
    for (int i = 0; i < N; i++) c[i] = -a[i];
    return c;
}

template<typename Dtype, int N>
ArrayWithOp<Dtype, N> operator*(const ArrayWithOp<Dtype, N>& a, double s)
{
    ArrayWithOp<Dtype, N> c;
    for (int i = 0; i < N; i++) c[i] = a[i]*s;
    return c;
}

template<typename Dtype, int N>
ArrayWithOp<Dtype, N> operator*(double s, const ArrayWithOp<Dtype, N>& a)
{
    ArrayWithOp<Dtype, N> c;
    for (int i = 0; i < N; i++) c[i] = a[i]*s;
    return c;
}

template<typename Dtype, int N>
ArrayWithOp<Dtype, N> operator/(const ArrayWithOp<Dtype, N>& a, double s)
{
    ArrayWithOp<Dtype, N> c;
    for (int i = 0; i < N; i++) c[i] = a[i]/s;
    return c;
}


/**
 * We use the same name `Jet` as in ceres-solver.
 */
template<typename Dtype, int N>
class Jet
{
 public:
    Jet() : a_()
    {
        v_.fill(0);
    }

    /** conversion from a scalar */
    Jet(const Dtype& a)     // NOLINT
    {
        a_ = a;
        v_.fill(0);
    }

    Jet(const Dtype& a, int i, Dtype value = 1)
    {
        a_ = a;
        v_.fill(0);
        v_.at(i) = value;
    }

    void set(const Dtype& a, int i, Dtype value = 1)
    {
        a_ = a;
        v_.fill(0);
        v_.at(i) = value;
    }

    std::string to_string() const
    {
        std::ostringstream ss;
        ss << "[ " << a_ << ", ";
        for (int i = 0; i < N; i++) ss << v_[i] << " ";
        ss << "]";
        ss << "\n";
        return ss.str();
    }

    Jet& operator +=(const Jet& f) {return *this = *this + f;}
    Jet& operator -=(const Jet& f) {return *this = *this - f;}
    Jet& operator *=(const Jet& f) {return *this = *this * f;}
    Jet& operator /=(const Jet& f) {return *this = *this / f;}

    Jet& operator +=(const Dtype& s) {return *this = *this + s;}
    Jet& operator -=(const Dtype& s) {return *this = *this - s;}
    Jet& operator *=(const Dtype& s) {return *this = *this * s;}
    Jet& operator /=(const Dtype& s) {return *this = *this / s;}

    operator Dtype() const
    {
        return a_;
    }

    Dtype a_;                   //!< value
    ArrayWithOp<Dtype, N> v_;   //!< gradient
};

template<typename Dtype, int N>
std::ostream& operator << (std::ostream& os, const Jet<Dtype, N>& f)
{
    os << f.to_string();
    return os;
}


//----------------------------------------
//  negate
//  -
//----------------------------------------
template<typename Dtype, int N>
Jet<Dtype, N> operator - (const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res;
    res.a_ = -f.a_;
    res.v_ = -f.v_;
    return res;
}

//----------------------------------------
//  scalars
//  +, -, *, /
//----------------------------------------

template<typename Dtype, int N>
Jet<Dtype, N> operator + (const Jet<Dtype, N>& f, Dtype s)
{
    Jet<Dtype, N> res(f);
    res.a_ += s;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator + (Dtype s, const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res(f);
    res.a_ += s;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator - (const Jet<Dtype, N>& f, Dtype s)
{
    Jet<Dtype, N> res(f);
    res.a_ -= s;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator - (Dtype s, const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res;
    res.a_ = s - f.a_;
    res.v_ = -f.v_;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator * (const Jet<Dtype, N>& f, Dtype s)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ * s;
    res.v_ = f.v_ * s;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator * (Dtype s, const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ * s;
    res.v_ = f.v_ * s;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator / (const Jet<Dtype, N>& f, Dtype s)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ / s;
    res.v_ = f.v_ / s;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator / (Dtype s, const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res;
    res.a_ = s / f.a_;
    res.v_ = - s * f.v_ / (f.a_ * f.a_);
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator + (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ + g.a_;
    res.v_ = f.v_ + g.v_;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator - (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ - g.a_;
    res.v_ = f.v_ - g.v_;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator * (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ * g.a_;
    res.v_ = f.a_*g.v_ + g.a_*f.v_;
    return res;
}

template<typename Dtype, int N>
Jet<Dtype, N> operator / (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{
    Jet<Dtype, N> res;
    res.a_ = f.a_ / g.a_;
    res.v_ = f.v_ / g.a_ - (f.a_/(g.a_*g.a_))*g.v_;
    return res;
}

template<typename Dtype, int N>
bool operator == (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{return f.a_ == g.a_;}

template<typename Dtype, int N>
bool operator != (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{return f.a_ != g.a_;}

template<typename Dtype, int N>
bool operator < (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{return f.a_ < g.a_;}

template<typename Dtype, int N>
bool operator <= (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{return f.a_ <= g.a_;}

template<typename Dtype, int N>
bool operator > (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{return f.a_ > g.a_;}

template<typename Dtype, int N>
bool operator >= (const Jet<Dtype, N>& f, const Jet<Dtype, N>& g)
{return f.a_ >= g.a_;}


//--------------------
//  math functions
//      std::exp
//      std::log
//      std::max
//--------------------
using std::max;

// do not return a reference here!
template<typename Dtype, int N>
Jet<Dtype, N> max(const Jet<Dtype, N>& f, const Jet<Dtype, N>&g)
{
    return (f < g) ? g : f;
}

using std::exp;

template<typename Dtype, int N>
Jet<Dtype, N> exp(const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res;

    auto s = exp(f.a_);
    res.a_ = s;
    res.v_ = s * f.v_;

    return res;
}

using std::log;

template<typename Dtype, int N>
Jet<Dtype, N> log(const Jet<Dtype, N>& f)
{
    Jet<Dtype, N> res;

    auto s = Dtype(1)/f.a_;
    res.a_ = log(f.a_);
    res.v_ = s * f.v_;

    return res;
}

}  // namespace cnn

