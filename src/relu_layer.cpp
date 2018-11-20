#include <glog/logging.h>

#include <vector>

#include "cnn/jet.hpp"
#include "cnn/relu_layer.hpp"

namespace cnn
{

template<typename Dtype>
ReLULayer<Dtype>::ReLULayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{}

template<typename Dtype>
void ReLULayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& top_gradient)
{
    CHECK_EQ(bottom.size(), 1) << "relu accepts only 1 input";
    CHECK_EQ(top.size(), 1) << "relu generates only 1 output";

    top[0]->init_like(*bottom[0]);

    if (this->proto_.phase() == TRAIN)
    {
        CHECK_EQ(bottom_gradient.size(), 1);

        if (!bottom_gradient[0]->has_same_shape(*bottom[0]))
        {
            bottom_gradient[0]->init_like(*bottom[0]);
        }

        CHECK_EQ(top_gradient.size(), 1);
        top_gradient[0]->init_like(*top[0]);
    }
}

template<typename Dtype>
void ReLULayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
    const auto& b = *bottom[0];
    auto& t = *top[0];
    for (int i = 0; i < b.total_; i++)
    {
        t[i] = max(b[i], Dtype(0));     // NOLINT
    }
}

template<typename Dtype>
void ReLULayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& /*top*/,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
    const auto& b = *bottom[0];
    auto& bg = *bottom_gradient[0];

    const auto& tg = *top_gradient[0];

    for (int i = 0; i < b.total_; i++)
    {
        bg[i] = tg[i]*(b[i] >= Dtype(0));
    }
}

}  // namespace cnn


