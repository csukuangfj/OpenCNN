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
            std::vector<Array<Dtype>*>* top) override;

    void fprop(
            const std::vector<const Array<Dtype>*>& input,
            std::vector<Array<Dtype>*>* output) override;

    void bprop(
            const std::vector<const Array<Dtype>*>&,
            std::vector<const Array<Dtype>*>*,
            const std::vector<const Array<Dtype>*>&,
            const std::vector<const Array<Dtype>*>&) override
    {}
 private:
    int n_;
    int c_;
    int h_;
    int w_;
    bool has_label_;
};

}  // namespace cnn
