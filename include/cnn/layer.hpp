#pragma once

#include <vector>

#include "proto/cnn.pb.h"

#include "cnn/array.hpp"
#include "cnn/array_math.hpp"

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

    std::vector<Array<Dtype>*> mutable_param()
    {
        std::vector<Array<Dtype>*> res;
        for (int i = 0; i < param_.size(); i++)
        {
            res.push_back(param_[i].get());
        }
        return res;
    }

    std::vector<const Array<Dtype>*> param() const
    {
        std::vector<const Array<Dtype>*> res;
        for (int i = 0; i < param_.size(); i++)
        {
            res.push_back(param_[i].get());
        }
        return res;
    }

    std::vector<Array<Dtype>*> mutable_gradient()
    {
        std::vector<Array<Dtype>*> res;
        for (int i = 0; i < gradient_.size(); i++)
        {
            res.push_back(gradient_[i].get());
        }
        return res;
    }

    std::vector<const Array<Dtype>*> gradient() const
    {
        std::vector<const Array<Dtype>*> res;
        for (int i = 0; i < gradient_.size(); i++)
        {
            res.push_back(gradient_[i].get());
        }
        return res;
    }

    void clear_gradient()
    {
        for (auto& g : gradient_)
        {
            if (g)
            {
                set_to<Dtype>(g.get(), 0);
            }
        }
    }

    /**
     * At layer construction, we have no idea of the shape of its inputs,
     * so this function MUST be called after constructing the whole network.
     */
    virtual void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& bottom_gradient,
            const std::vector<Array<Dtype>*>& top,
            const std::vector<Array<Dtype>*>& top_gradient) = 0;

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

#include "../../src/layer.cpp"
