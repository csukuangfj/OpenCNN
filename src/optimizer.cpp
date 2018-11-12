
#include <glog/logging.h>

#include <string>

#include "cnn/io.hpp"
#include "cnn/optimizer.hpp"

namespace cnn
{
template<typename Dtype>
Optimizer<Dtype>::Optimizer(const OptimizerProto& _proto)
{
    init(_proto);
}

template<typename Dtype>
Optimizer<Dtype>::Optimizer(const std::string& filename)
{
    OptimizerProto _proto;
    read_proto_txt(filename, &_proto);
    init(_proto);
}

template<typename Dtype>
void Optimizer<Dtype>::init(const OptimizerProto& _proto)
{
    proto_ = _proto;
    LOG(INFO) << "\n\n" << proto_.DebugString() << "\n";

    auto network_filename = proto_.model_filename();
    network_.reset(new Network<Dtype>(network_filename));
    LOG(INFO) << "\n\n--\n" << network_->proto().DebugString() << "\n---\n";
}

template class Optimizer<float>;
template class Optimizer<double>;

}  // namespace cnn
