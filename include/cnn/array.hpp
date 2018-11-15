#pragma once

#include <string>
#include <vector>

#include "proto/cnn.pb.h"

namespace cnn
{

template<typename Dtype>
class Array
{
 public:
    Array();
    ~Array();
    Array(const Array<Dtype>&) = delete;
    Array& operator=(const Array<Dtype>&) = delete;

    void init(int n, int c, int h, int w);
    void init_like(const Array<Dtype> &arr);

    bool has_same_shape(const Array<Dtype> &arr) const;
    bool has_same_shape(const std::vector<int>& vec) const;
    std::vector<int> shape_vec() const {return {n_, c_, h_, w_};}

    /**
     * Return the element at n*c_*h_*w_ + c*h_*w_ + h*w_ + w,
     * i.e., (n*c_ + c)*h_*w_ + h*w_ + w,
     * i.e., ((n*c_ + c)*h_ + h)*w_ + w
     */
    const Dtype& at(int n, int c, int h, int w) const;
    Dtype& at(int n, int c, int h, int w);

    // no range check
    const Dtype& operator()(int n, int c, int h, int w) const;
    Dtype& operator()(int n, int c, int h, int w);

    const Dtype& operator[](int i) const;
    Dtype& operator[](int i);

    std::string shape_info() const;

    int n_;      //!< number of batches
    int c_;      //!< number of channels
    int h_;      //!< image height, i.e., number of rows
    int w_;      //!< image width, number of columns
    int total_;  //!< n_*c_*h_*w_, number of elements

    Dtype* d_;  //!< pointer to the data

 public:
    void from_proto(const ArrayProto& proto);
    void to_proto(ArrayProto* proto) const;
};

}  // namespace cnn

#include "../../src/array.cpp"
