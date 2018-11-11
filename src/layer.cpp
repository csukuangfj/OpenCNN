#include <glog/logging.h>

#include "cnn/full_connected_layer.hpp"
#include "cnn/input_layer.hpp"
#include "cnn/layer.hpp"

namespace cnn
{
template<typename Dtype>
Layer<Dtype>::Layer(const LayerProto& _proto)
    : param_(),
      proto_(_proto)
{
    if (proto_.param().d_size())
    {
        param_.from_proto(proto_.param());
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

