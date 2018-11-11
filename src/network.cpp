
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
    // a network MUST have at least two layers: an input layer and an output layer
    //
    CHECK_GE(proto_.layer_proto_size(), 2);

    const auto& input_layer = proto_.layer_proto(0);
    CHECK_EQ(input_layer.type(), INPUT)
        << "The 0th layer has to be of type INPUT!";

    CHECK_EQ(input_layer.bottom_size(), 0)
        << "Input layer should have no bottom!";

    // allocate space for the input
    for (int i = 0; i < input_layer.top_size(); i++)
    {
        auto d = std::make_shared<Array<Dtype>>();
        add_data(input_layer.top(i), d);
    }

    layers_.push_back(Layer<Dtype>::create(input_layer));

    // create other layers
    for (int i = 1; i < proto_.layer_proto_size(); i++)
    {
        // check its bottom has been created!
        const auto &layer_proto = proto_.layer_proto(i);
        for (int j = 0; j < layer_proto.bottom_size(); j++)
        {
            const auto &name = layer_proto.bottom(j);
            CHECK_EQ(data_.count(name), 1)
                << "bottom with name " << name
                << " does not exist!";
        }

        // then creates its top
        for (int j = 0; j < layer_proto.top_size(); j++)
        {
            auto d = std::make_shared<Array<Dtype>>();
            add_data(layer_proto.top(j), d);
        }

        layers_.push_back(Layer<Dtype>::create(layer_proto));
    }
}


template<typename Dtype>
void Network<Dtype>::reshape()
{
    static bool init = false;
    if (false)
    {
        LOG(FATAL) << "reshape() should only be called once!";
    }
    init = true;

    layers_[0]->reshape({}, get_data_output(0));
    for (int i = 1; i < layers_.size(); i++)
    {
        layers_[i]->reshape(get_data_input(i), get_data_output(i));
    }
}

template<typename Dtype>
void Network<Dtype>::fprop()
{

}

template<typename Dtype>
void Network<Dtype>::add_data(
        const std::string& name,
        std::shared_ptr<Array<Dtype>> arr)
{
    CHECK_EQ(data_.count(name), 0)
        << "duplicate name " << name;
    data_[name] = arr;
    LOG(INFO) << "add: " << name;
}

template<typename Dtype>
std::vector<const Array<Dtype>*>
Network<Dtype>::get_data_input(int i)
{
    std::vector<const Array<Dtype>*> res;
    const auto &layer_proto = layers_[i]->proto();
    int n = layer_proto.bottom_size();
    for (int i = 0; i < n; i++)
    {
        const auto &name = layer_proto.bottom(i);
        CHECK_EQ(data_.count(name), 1)
            << "data with name " << name << " does not exist!";
        res.push_back(data_[name].get());
    }
    return res;
}

template<typename Dtype>
std::vector<Array<Dtype>*>
Network<Dtype>::get_data_output(int i)
{
    std::vector<Array<Dtype>*> res;
    const auto &layer_proto = layers_[i]->proto();
    int n = layer_proto.top_size();
    for (int i = 0; i < n; i++)
    {
        const auto &name = layer_proto.top(i);
        CHECK_EQ(data_.count(name), 1)
            << "data with name " << name << " does not exist!";
        res.push_back(data_[name].get());
    }
    return res;
}

template class Network<float>;
template class Network<double>;

}  // namespace cnn
