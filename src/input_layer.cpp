#include <glog/logging.h>

#include <vector>

#include "cnn/input_layer.hpp"

namespace cnn
{
template<typename Dtype>
InputLayer<Dtype>::InputLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    const auto param = _proto.input_param();
    n_ = param.n();
    c_ = param.c();
    h_ = param.h();
    w_ = param.w();
    has_label_ = param.has_label();
}

template<typename Dtype>
void InputLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& /*input*/,
        std::vector<Array<Dtype>*>* output)
{
    if (has_label_)
    {
        CHECK_EQ(output->size(), 2);
    }
    else    // NOLINT
    {
        CHECK_EQ(output->size(), 1);
    }

    output[0][0]->init(n_, c_, h_, w_);
}

template<typename Dtype>
void InputLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& input,
        std::vector<Array<Dtype>*>* output)
{
    LOG(INFO) << "fprop in input layer!";
}


template class InputLayer<float>;
template class InputLayer<double>;


}  // namespace cnn

