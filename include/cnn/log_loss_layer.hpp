#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{
/**
 * For classification only!
 *
 * It accepts two inputs: bottom[0] and bottom[1].
 *
 * bottom[0] is the prediction with shape (N, C, H, W).
 * The meaning of the shape can be interpreted as follows:
 * (1) N is the batch size. (2) H and W can be considered
 * as the size of an image. (3) C represents the number of channels
 * of the image; in the case of multi-class classification, C
 * is the number of classes we have and the meaning of each pixel
 * in the channel k indicates the probability that this pixel belongs
 * to class k.
 *
 * The user has to ensure that the pixel values are in
 * the range of [0, 1] and it is normalized, i.e., sum to 1.
 *
 * This layer lays usually on top of the softmax layer.
 *
 * bottom[1] is the ground truth and has the shape (N, 1, H, W).
 * values of every element in bottom[1] have to be an integer
 * in the range [0, C-1].
 */
template<typename Dtype>
class LogLossLayer : public Layer<Dtype>
{
 public:
    explicit LogLossLayer(const LayerProto&);

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
    Dtype loss_;
};

}  // namespace cnn

#include "../../src/log_loss_layer.cpp"
