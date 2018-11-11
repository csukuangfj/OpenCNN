#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{
template<typename Dtype>
class FullConnectedLayer : public Layer<Dtype>
{
 public:
    explicit FullConnectedLayer(const LayerProto&);

    void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            std::vector<Array<Dtype>*>* top) override;

    void fprop(
            const std::vector<const Array<Dtype>*>& input,
            std::vector<Array<Dtype>*>* output) override;

    void bprop(
            const std::vector<const Array<Dtype>*>& bottom,
            std::vector<const Array<Dtype>*>* bottom_gradient,
            const std::vector<const Array<Dtype>*>& top,
            const std::vector<const Array<Dtype>*>& top_gradient) override;
};

}  // namespace cnn
