#include <glog/logging.h>

#include "cnn/full_connected_layer.hpp"
#include "cnn/input_layer.hpp"
#include "cnn/l2_loss_layer.hpp"
#include "cnn/layer.hpp"

namespace cnn
{
template<typename Dtype>
Layer<Dtype>::Layer(const LayerProto& _proto)
    : param_(),
      proto_(_proto)
{
    if (proto_.param_size())
    {
        param_.resize(proto_.param_size());
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
    std::shared_ptr<Layer<Dtype>> res;
    switch (_proto.type())
    {
        case INPUT:
            res.reset(new InputLayer<Dtype>(_proto));
            break;
        case FULL_CONNECTED:
            res.reset(new FullConnectedLayer<Dtype>(_proto));
            break;
        case L2_LOSS:
            res.reset(new L2LossLayer<Dtype>(_proto));
            break;
        default:
            LOG(FATAL) << "Unknown layer type: "
                << LayerType_Name(_proto.type());
            break;
    }
    CHECK_NOTNULL(res.get());
    return res;
}

template class Layer<float>;
template class Layer<double>;

}  // namespace cnn

