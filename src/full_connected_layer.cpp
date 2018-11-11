#include <glog/logging.h>

#include <vector>

#include "cnn/full_connected_layer.hpp"

namespace cnn
{
template<typename Dtype>
FullConnectedLayer<Dtype>::FullConnectedLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    num_output_ = _proto.fc_proto().num_output();
}


template<typename Dtype>
void FullConnectedLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    LOG(INFO) << "fc reshape";
    CHECK_EQ(input.size(), 1);

    // resize output
    CHECK_EQ(output.size(), 1);
    int n = input[0]->n_;
    int c = num_output_;
    int h = 1;
    int w = 1;
    output[0]->init(n, c, h, w);
}

template<typename Dtype>
void FullConnectedLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    LOG(INFO) << "fprop in fc!";
}

template<typename Dtype>
void FullConnectedLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

template class FullConnectedLayer<float>;
template class FullConnectedLayer<double>;

}  // namespace cnn

