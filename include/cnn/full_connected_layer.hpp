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
    int num_output_;
};

}  // namespace cnn
