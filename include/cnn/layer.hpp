#pragma once

#include <vector>

#include "proto/cnn.pb.h"

#include "cnn/array.hpp"

namespace cnn
{
/**
 *
 * Every layer MUST implement the following functions
 *  * reshape
 *  * fprop
 *  * bprop
 */
template<typename Dtype>
class Layer
{
 public:
    explicit Layer(const LayerProto&);
    static std::shared_ptr<Layer<Dtype>> create(const LayerProto&);

    const LayerProto& proto() const {return proto_;}
    LayerProto& proto() {return proto_;}

    /**
     * At layer construction, we have no idea of the shape of its inputs,
     * so this function MUST be called after constructing the whole network.
     */
    virtual void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            std::vector<Array<Dtype>*>* top) = 0;

    /**
     * forward propagation
     */
    virtual void fprop(
            const std::vector<const Array<Dtype>*>& bottom,
            std::vector<Array<Dtype>*>* top) = 0;

    /**
     * backward propagation
     */
    virtual void bprop(
            const std::vector<const Array<Dtype>*>& bottom,
            std::vector<const Array<Dtype>*>* bottom_gradient,
            const std::vector<const Array<Dtype>*>& top,
            const std::vector<const Array<Dtype>*>& top_gradient) = 0;
 protected:
    Array<Dtype> param_;
    Array<Dtype> gradient_;
 private:
    LayerProto proto_;

 private:
    Layer(const Layer<Dtype>&) = delete;
    Layer& operator=(const Layer<Dtype>&) = delete;
};

}  // namespace cnn