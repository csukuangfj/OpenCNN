#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{

template<typename Dtype>
class InputLayer : public Layer<Dtype>
{
 public:
    explicit InputLayer(const LayerProto&);

    void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& bottom_gradient,
            const std::vector<Array<Dtype>*>& top,
            const std::vector<Array<Dtype>*>& top_gradient) override;

    void fprop(
            const std::vector<const Array<Dtype>*>& input,
            const std::vector<Array<Dtype>*>& output) override;

    void bprop(
            const std::vector<const Array<Dtype>*>&,
            const std::vector<Array<Dtype>*>&,
            const std::vector<const Array<Dtype>*>&,
            const std::vector<const Array<Dtype>*>&) override
    {}
 private:
    int n_;
    int c_;
    int h_;
    int w_;
};

}  // namespace cnn
