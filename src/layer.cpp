#include <glog/logging.h>

#include "cnn/convolution_layer.hpp"
#include "cnn/drop_out_layer.hpp"
#include "cnn/full_connected_layer.hpp"
#include "cnn/input_layer.hpp"
#include "cnn/jet.hpp"
#include "cnn/l2_loss_layer.hpp"
#include "cnn/layer.hpp"
#include "cnn/log_loss_layer.hpp"
#include "cnn/max_pooling_layer.hpp"
#include "cnn/relu_layer.hpp"
#include "cnn/softmax_layer.hpp"
#include "cnn/softmax_with_log_loss_layer.hpp"

namespace cnn
{
template<typename Dtype>
Layer<Dtype>::Layer(const LayerProto& _proto)
    : param_(),
      proto_(_proto)
{
    if (proto_.param_size())
    {
        param_.clear();
        for (int i = 0; i < proto_.param_size(); i++)
        {
            auto arr = std::make_shared<Array<Dtype>>();
            arr->from_proto(proto_.param(i));
            param_.push_back(arr);
        }
    }
}

template<typename Dtype>
std::shared_ptr<Layer<Dtype>>
Layer<Dtype>::create(const LayerProto& _proto)
{
#define CREATE_LAYER(type_name, class_name)         \
    case type_name:                                 \
        res.reset(new class_name<Dtype>(_proto));   \
        break

    std::shared_ptr<Layer<Dtype>> res;
    switch (_proto.type())
    {
        CREATE_LAYER(INPUT, InputLayer);
        CREATE_LAYER(FULL_CONNECTED, FullConnectedLayer);
        CREATE_LAYER(L2_LOSS, L2LossLayer);
        CREATE_LAYER(SOFTMAX, SoftmaxLayer);
        CREATE_LAYER(LOG_LOSS, LogLossLayer);
        CREATE_LAYER(SOFTMAX_WITH_LOG_LOSS, SoftmaxWithLogLossLayer);
        CREATE_LAYER(CONVOLUTION, ConvolutionLayer);
        CREATE_LAYER(RELU, ReLULayer);
        CREATE_LAYER(MAX_POOLING, MaxPoolingLayer);
        CREATE_LAYER(DROP_OUT, DropoutLayer);

        default:
            LOG(FATAL) << "Unknown layer type: "
                << LayerType_Name(_proto.type());
            break;
    }

#undef CREATE_LAYER

    CHECK_NOTNULL(res.get());
    return res;
}

template<typename Dtype>
void Layer<Dtype>::copy_trained_layer(const LayerProto& p)
{
    CHECK_EQ(proto_.name(), p.name());
    CHECK_EQ(proto_.type(), p.type());
    CHECK_EQ(proto_.top_size(), p.top_size());
    CHECK_EQ(proto_.bottom_size(), p.bottom_size());

    if (param_.size())
    {
        CHECK_EQ(param_.size(), p.param_size());
        param_.clear();
    }

    for (int i = 0; i < p.param_size(); i++)
    {
        auto arr = std::make_shared<Array<Dtype>>();
        arr->from_proto(p.param(i));
        param_.push_back(arr);
    }
}

}  // namespace cnn

