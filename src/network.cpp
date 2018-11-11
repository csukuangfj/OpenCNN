
#include <glog/logging.h>

#include "cnn/io.hpp"
#include "cnn/network.hpp"

namespace cnn
{
template<typename Dtype>
Network<Dtype>::Network(const NetworkProto &_proto)
{
    init(_proto);
}

template<typename Dtype>
Network<Dtype>::Network(const std::string &filename,
        bool is_binary /*= false*/)
{
    init(filename, is_binary);
}

template<typename Dtype>
void Network<Dtype>::init(const std::string &filename,
        bool is_binary /*= false*/)
{
    NetworkProto network_proto;
    if (is_binary)
    {
        read_proto_bin(filename, &network_proto);
    }
    else    // NOLINT
    {
        read_proto_txt(filename, &network_proto);
    }

    // LOG(INFO) << layer_proto.DebugString();
    init(network_proto);
}

template<typename Dtype>
void Network<Dtype>::init(const NetworkProto& _proto)
{
    proto_ = _proto;
    LOG(INFO) << "\n" << proto_.DebugString();
}


template class Network<float>;
template class Network<double>;

}  // namespace cnn
