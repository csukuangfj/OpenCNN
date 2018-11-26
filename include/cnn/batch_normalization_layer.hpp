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
 *
 * Feature maps, i.e., channels, are normalized over batches.
 *
 * @code
 *      for c = 0:num_channels-1
 *          total = 0
 *          for n = 0:num_batches-1
 *              add all pixels in channel c, batch n
 *              accumulate the result in total
 *
 *          average total
 *
 *          for n = 0:num_batches-1
 *              subtract total from all pixels in channel c, batch n
 *
 *          total = 0
 *          for n = 0:num_batches-1
 *              add the square of all pixels in channel c, batch n
 *              accumulate the result in total
 *
 *          average total
 *          stddev = sqrt(total + eps)
 *          for n = 0:num_batches-1
 *               divided by stddev of all pixels in channel c, batch n
 * @endcode
 *
 * param[0]: channel scale with shape (1, C, 1, 1)
 * param[1]: channel bias with shape (1, C, 1, 1)
 * param[2]: channel mean with shape (1, C, 1, 1)
 * param[3]: channel stddev with shape (1, C, 1, 1)
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

 private:
    /** avoid dividing by 0 */
    Dtype eps_ = 1e-5;

    /** moving_mean = moving_mean*momentum + mini_batch_mean*(1-momentum) */
    Dtype momentum_;

    // the following variables are used only in the train phase
    Array<Dtype> x_minus_mu_;   //!< saves x - mini_batch_mean
    Array<Dtype> mu_;           //!< mini_batch_mean
    Array<Dtype> var_;          //!< mini_batch_variance
};

}  // namespace cnn

#include "../../src/batch_normalization_layer.cpp"


