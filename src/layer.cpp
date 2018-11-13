#include <glog/logging.h>

#include "cnn/full_connected_layer.hpp"
#include "cnn/input_layer.hpp"
#include "cnn/l2_loss_layer.hpp"
#include "cnn/layer.hpp"
#include "cnn/log_loss_layer.hpp"
#include "cnn/softmax_layer.hpp"

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

        default:
            LOG(FATAL) << "Unknown layer type: "
                << LayerType_Name(_proto.type());
            break;
    }

#undef CREATE_LAYER

    CHECK_NOTNULL(res.get());
    return res;
}

template class Layer<float>;
template class Layer<double>;

}  // namespace cnn

