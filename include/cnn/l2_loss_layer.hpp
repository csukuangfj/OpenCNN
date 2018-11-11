#pragma once

#include <vector>

#include "cnn/layer.hpp"

namespace cnn
{
/**
 *
 * for regression only!
 */
template<typename Dtype>
class L2LossLayer : public Layer<Dtype>
{
 public:
    explicit L2LossLayer(const LayerProto&);

    void reshape(
            const std::vector<const Array<Dtype>*>& bottom,
            const std::vector<Array<Dtype>*>& top) override;

    void fprop(
            const std::vector<const Array<Dtype>*>& input,
            const std::vector<Array<Dtype>*>& output) override;

    void bprop(
            const std::vector<const Array<Dtype>*>&,
            const std::vector<Array<Dtype>*>&,
            const std::vector<const Array<Dtype>*>&,
            const std::vector<const Array<Dtype>*>&) override;
 private:
    Dtype loss_;
};

}  // namespace cnn
