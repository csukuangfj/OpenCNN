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

    std::vector<std::shared_ptr<Array<Dtype>>>& param()
    {return param_;}

    const std::vector<std::shared_ptr<Array<Dtype>>>& param() const
    {return param_;}

    //! @todo disable gradient memory allocation in the test phase
    std::vector<Array<Dtype>*> mutable_gradient()
    {
        std::vector<Array<Dtype>*> res;
        for (int i = 0; i < gradient_.size(); i++)
        {
            res.push_back(gradient_[i].get());
        }
        return res;
    }

    std::vector<const Array<Dtype>*> get_gradient() const
    {
        std::vector<const Array<Dtype>*> res;
        for (int i = 0; i < gradient_.size(); i++)
        {
            res.push_back(gradient_[i].get());
        }
        return res;
    }

    /**
     * At layer construction, we have no idea of the shape of its inputs,
     * so this function MUST be called after constructing the whole network.
     */
    virtual void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& top) = 0;

    /**
     * forward propagation
     */
    virtual void fprop(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& top) = 0;

    /**
     * backward propagation
     */
    virtual void bprop(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& bottom_gradient,
            const std::vector<const Array<Dtype>*>& top,
            const std::vector<const Array<Dtype>*>& top_gradient) = 0;
 protected:
    std::vector<std::shared_ptr<Array<Dtype>>> param_;
    std::vector<std::shared_ptr<Array<Dtype>>> gradient_;

    LayerProto proto_;

 private:
    Layer(const Layer<Dtype>&) = delete;
    Layer& operator=(const Layer<Dtype>&) = delete;
};

}  // namespace cnn
