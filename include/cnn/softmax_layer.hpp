#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{

/**
 * Compute softmax over channels.
 *
 * For example, for a color image with 3 channels, we compute softmax
 * for every pixel over its r, g, b channels such that
 * (r_i, g_i, b_i) = softmax(r_i, g_i, b_i), where
 * r_i is the red channel for pixel i and so on for g_i and b_i;
 *
 * softmax(a, b, c) = (e^a/(e^a+ e^b + e^c), e^b/(e^a+e^b+e^c), e^c/(e^a+e^b+e^c))
 *
 * if a = max(a, b, c), then
 * softmax(a, b, c) = softmax(a-a, b-a, c-a) = softmax(0, b-a, c-a)
 */
template<typename Dtype>
class SoftmaxLayer : public Layer<Dtype>
{
 public:
    explicit SoftmaxLayer(const LayerProto&);

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
 private:
    Array<Dtype> buffer_;
};

}  // namespace cnn

#include "../../src/softmax_layer.cpp"
