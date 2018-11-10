#include <glog/logging.h>

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
void Array<Dtype>::init(int n, int c, int h, int w)
{
    CHECK_GT(n, 0);
    CHECK_GT(c, 0);
    CHECK_GT(h, 0);
    CHECK_GT(w, 0);

    int total = n*c*h*w;
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
Dtype Array<Dtype>::at(int n, int c, int h, int w) const
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
Dtype Array<Dtype>::operator()(int n, int c, int h, int w) const
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
Dtype Array<Dtype>::operator[](int i) const
{
    return d_[i];
}

template<typename Dtype>
Dtype& Array<Dtype>::operator[](int i)
{
    return d_[i];
}


template class Array<float>;
template class Array<double>;

}  // namespace cnn
