#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{
/**
 * It has one bottom and one top.
 *
 * bottom[0] has shape (N, C, H, W)
 *
 * top[0] has shape (N, C, h, w)
 *
 * where
 *  h = (H - win_size)/stride + 1
 *  w = (W - win_size)/stride + 1
 */
template<typename Dtype>
class MaxPoolingLayer : public Layer<Dtype>
{
 public:
    explicit MaxPoolingLayer(const LayerProto&);

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
    int win_size_;
    int stride_;
};

}  // namespace cnn

#include "../../src/max_pooling_layer.cpp"


