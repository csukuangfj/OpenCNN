#include <glog/logging.h>

#include <sstream>
#include <string>

#include "cnn/array.hpp"

namespace cnn
{

template<typename Dtype>
Array<Dtype>::Array()
    : n_(0),
      c_(0),
      h_(0),
      w_(0),
      total_(0),
      d_(nullptr)
{}

template<typename Dtype>
Array<Dtype>::~Array()
{
    if (d_) delete[] d_;
}

template<typename Dtype>
void Array<Dtype>::init_like(const Array<Dtype> &arr)
{
    if (this == &arr) return;
    init(arr.n_, arr.c_, arr.h_, arr.w_);
}

template<typename Dtype>
void Array<Dtype>::init(int n, int c, int h, int w)
{
    CHECK_GE(n, 0);
    CHECK_GE(c, 0);
    CHECK_GE(h, 0);
    CHECK_GE(w, 0);

    int total = n*c*h*w;
    if (total == 0)
    {
        if (d_)
        {
            delete[] d_;
            d_ = nullptr;
        }
        n_ = c_ = h_ = w_ = 0;
        total_ = 0;
        return;
    }

    if (total != total_)
    {
        if (d_) delete[] d_;

        d_ = new Dtype[n*c*h*w];
    }

    memset(d_, 0, total*sizeof(Dtype));
    n_ = n;
    c_ = c;
    h_ = h;
    w_ = w;
    total_ = total;
}

template<typename Dtype>
const Dtype& Array<Dtype>::at(int n, int c, int h, int w) const
{
    CHECK_GE(n, 0);
    CHECK_LT(n, n_);
    CHECK_GE(c, 0);
    CHECK_LT(c, c_);
    CHECK_GE(h, 0);
    CHECK_LT(h, h_);
    CHECK_GE(w, 0);
    CHECK_LT(w, w_);
    int i = ((n*c_ + c)*h_ + h)*w_ + w;
    return d_[i];
}

template<typename Dtype>
Dtype& Array<Dtype>::at(int n, int c, int h, int w)
{
    CHECK_GE(n, 0);
    CHECK_LT(n, n_);
    CHECK_GE(c, 0);
    CHECK_LT(c, c_);
    CHECK_GE(h, 0);
    CHECK_LT(h, h_);
    CHECK_GE(w, 0);
    CHECK_LT(w, w_);
    int i = ((n*c_ + c)*h_ + h)*w_ + w;
    return d_[i];
}

template<typename Dtype>
const Dtype& Array<Dtype>::operator()(int n, int c, int h, int w) const
{
    int i = ((n*c_ + c)*h_ + h)*w_ + w;
    return d_[i];
}

template<typename Dtype>
Dtype& Array<Dtype>::operator()(int n, int c, int h, int w)
{
    int i = ((n*c_ + c)*h_ + h)*w_ + w;
    return d_[i];
}

template<typename Dtype>
const Dtype& Array<Dtype>::operator[](int i) const
{
    return d_[i];
}

template<typename Dtype>
Dtype& Array<Dtype>::operator[](int i)
{
    return d_[i];
}

template<typename Dtype>
std::string Array<Dtype>::shape_info() const
{
    std::ostringstream ss;
    ss << n_ << ", "
       << c_ << ", "
       << h_ << ", "
       << w_ << "\n";
    return ss.str();
}

template<typename Dtype>
void Array<Dtype>::from_proto(const ArrayProto& proto)
{
    init(proto.n(), proto.c(), proto.h(), proto.w());
    for (int i = 0; i < total_; i++)
    {
        d_[i] = static_cast<Dtype>(proto.d(i));
    }
}

template<typename Dtype>
void Array<Dtype>::to_proto(ArrayProto* proto)
{
    proto->set_n(n_);
    proto->set_c(c_);
    proto->set_h(h_);
    proto->set_w(w_);

    proto->clear_d();

    for (int i = 0; i < total_; i++)
    {
        proto->add_d(d_[i]);
    }
}

template class Array<float>;
template class Array<double>;

}  // namespace cnn
