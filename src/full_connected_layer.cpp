#include <glog/logging.h>

#include <vector>

#include "cnn/full_connected_layer.hpp"

namespace cnn
{
template<typename Dtype>
FullConnectedLayer<Dtype>::FullConnectedLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{}


template<typename Dtype>
void FullConnectedLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& input,
        std::vector<Array<Dtype>*>* output)
{
}

template<typename Dtype>
void FullConnectedLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& input,
        std::vector<Array<Dtype>*>* output)
{
    LOG(INFO) << "fprop in fc!";
}

template<typename Dtype>
void FullConnectedLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        std::vector<const Array<Dtype>*>* bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

template class FullConnectedLayer<float>;
template class FullConnectedLayer<double>;

}  // namespace cnn

