#include <glog/logging.h>

#include <vector>

#include "cnn/batch_normalization_layer.hpp"

namespace cnn
{

template<typename Dtype>
BatchNormalizationLayer<Dtype>::BatchNormalizationLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
}

template<typename Dtype>
void BatchNormalizationLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& top_gradient)
{
    CHECK_EQ(bottom.size(), 1) << "Batch normalziation accepts only 1 input";
    CHECK_EQ(top.size(), 1) << "Batch normalziation generates only 1 output";

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
void BatchNormalizationLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
}

template<typename Dtype>
void BatchNormalizationLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& /*bottom*/,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& /*top*/,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

}  // namespace cnn


