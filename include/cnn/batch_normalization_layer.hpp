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
 * Refer to the paper https://arxiv.org/pdf/1502.03167.pdf
 */
template<typename Dtype>
class BatchNormalizationLayer : public Layer<Dtype>
{
 public:
    explicit BatchNormalizationLayer(const LayerProto&);

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

#include "../../src/batch_normalization_layer.cpp"


