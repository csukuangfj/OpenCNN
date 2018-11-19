#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{
/**
 * It has one bottom and one top.
 *
 * Both bottom[0] and top[0] have shape (N, C, H, W).
 *
 * top[0]->d_[i] = max(0, bottom[0]->d_[i]);
 *
 */
template<typename Dtype>
class ReLULayer : public Layer<Dtype>
{
 public:
    explicit ReLULayer(const LayerProto&);

    void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& bottom_gradient,
            const std::vector<Array<Dtype>*>& top,
            const std::vector<Array<Dtype>*>& top_gradient) override;

    void fprop(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& top) override;

    void bprop(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& bottom_gradient,
            const std::vector<const Array<Dtype>*>& top,
            const std::vector<const Array<Dtype>*>& top_gradient) override;
};

}  // namespace cnn

#include "../../src/relu_layer.cpp"

