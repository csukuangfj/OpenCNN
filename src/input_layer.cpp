#include <glog/logging.h>

#include <vector>

#include "cnn/input_layer.hpp"

namespace cnn
{
template<typename Dtype>
InputLayer<Dtype>::InputLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    const auto param = _proto.input_proto();
    n_ = param.n();
    c_ = param.c();
    h_ = param.h();
    w_ = param.w();
}

template<typename Dtype>
void InputLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& /*bottom*/,
        const std::vector<Array<Dtype>*>& /*bottom_gradient*/,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& /*top_gradient*/)
{
    CHECK((top.size() == 1) || (top.size() == 2));

    top[0]->init(n_, c_, h_, w_);
    if (top.size() == 2)
    {
        // resize the label
        top[1]->init(n_, 1, 1, 1);
    }
}

template<typename Dtype>
void InputLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& /*input*/,
        const std::vector<Array<Dtype>*>& /*output*/)
{
    // TODO(fangjun) load data from here
}


template class InputLayer<float>;
template class InputLayer<double>;

}  // namespace cnn

