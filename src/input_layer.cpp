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
        const std::vector<const Array<Dtype>*>& /*input*/,
        const std::vector<Array<Dtype>*>& output)
{
    CHECK((output.size() == 1) || (output.size() == 2));

    output[0]->init(n_, c_, h_, w_);
    if (output.size() == 2)
    {
        // resize the label
        output[1]->init(n_, 1, 1, 1);
    }
}

template<typename Dtype>
void InputLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    LOG(INFO) << "fprop in input layer!";
}


template class InputLayer<float>;
template class InputLayer<double>;

}  // namespace cnn

