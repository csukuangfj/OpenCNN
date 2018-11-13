
#include <glog/logging.h>

#include <string>

#include "cnn/array_math.hpp"
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

    auto network_filename = proto_.model_filename();
    network_.reset(new Network<Dtype>(network_filename));
}

template<typename Dtype>
void Optimizer<Dtype>::start_training()
{
    int max_iter = proto_.max_iteration_num();
    network_->reshape();
    for (int i = 0; i < max_iter; i++)
    {
        network_->fprop();
        network_->bprop();
        update_parameters();
        if (i % 1000 == 0)
        {
            LOG(INFO) << "iteration: " << i;
            LOG(INFO) << "loss is: " << network_->get_loss();
            print_parameters();
        }
    }

    LOG(INFO) << "iteration: " << max_iter;
    LOG(INFO) << "loss is: " << network_->get_loss();
    print_parameters();
    network_->save_network("trained.prototxt");
}

template<typename Dtype>
void Optimizer<Dtype>::update_parameters()
{
    auto& layers = network_->layers();
    int num_layers = layers.size();

    Dtype learning_rate = proto_.learning_rate();
    // learning_rate /= network_->get_batch_size();

    // we skip the input layer since it has no parameters
    for (int i = 1; i < num_layers; i++)
    {
        auto param = layers[i]->mutable_param();
        auto gradient = layers[i]->gradient();
        for (int i = 0; i < param.size(); i++)
        {
            ax_plus_by<Dtype>(
                    param[i]->total_,
                    -learning_rate,
                    &gradient[i]->d_[0],
                    1,
                    &param[i]->d_[0]);
        }
    }
}

template<typename Dtype>
void Optimizer<Dtype>::print_parameters()
{
    auto& layers = network_->layers();
    int num_layers = layers.size();

    std::ostringstream ss;
    ss << "\n";
    ss << "batch size is: " << network_->get_batch_size() << "\n";
    // we skip the input layer since it has no parameters
    for (int i = 1; i < num_layers; i++)
    {
        auto param = layers[i]->mutable_param();
        if (param.empty()) continue;
        ss << "parameters for layer: " << layers[i]->proto().name();
        ss << "\n";

        for (int j = 0; j < param.size(); j++)
        {
            for (int k = 0; k < param[j]->total_; k++)
            {
                ss << param[j]->d_[k] << " ";
            }
            ss << "\n";
        }
    }

    LOG(INFO) << ss.str();
}

template class Optimizer<float>;
template class Optimizer<double>;

}  // namespace cnn
