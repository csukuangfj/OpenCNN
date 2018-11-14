
#include <glog/logging.h>

#include <fstream>
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
    std::ofstream of("abc.txt");
    int max_iter = proto_.max_iteration_num();
    network_->reshape();
    for (int i = 0; i < max_iter; i++)
    {
        network_->fprop();
        network_->bprop();
        update_parameters();
        LOG(INFO) << "loss is: " << network_->get_loss();
        of << i << "," << network_->get_loss() << "\n";
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

    std::ostringstream ss;
    // we skip the input layer since it has no parameters
    for (int i = 1; i < num_layers; i++)
    {
        auto param = layers[i]->mutable_param();
        auto gradient = layers[i]->gradient();

        for (int k = 0; k < param.size(); k++)
        {
            ss << "before update weight: \n";
            for (int m = 0; m < param[k]->total_; m++)
            {
                ss << param[k]->d_[m] << " ";
            }
            ss << "\n";

            ss << "gradient: \n";
            for (int m = 0; m < gradient[k]->total_; m++)
            {
                ss << -learning_rate*gradient[k]->d_[m] << " ";
            }
            ss << "\n";

            ax_plus_by<Dtype>(
                    param[k]->total_,
                    -learning_rate,
                    &gradient[k]->d_[0],
                    1,
                    &param[k]->d_[0]);
            ss << "after update: \n";
            for (int m = 0; m < param[k]->total_; m++)
            {
                ss << param[k]->d_[m] << " ";
            }
            ss << "\n";
        }
    }
    LOG(WARNING) << ss.str();
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
        ss << " gradient for layer: " << layers[i]->proto().name() << "\n";
        auto gradient = layers[i]->gradient();
        for (const auto &g : gradient)
        {
            if (g->total_)
            {
                for (int i = 0; i < g->total_; i++)
                {
                    ss << g->d_[i] << " ";
                }
            }
            ss << "\n";
        }
    }

    LOG(INFO) << ss.str();
}

template class Optimizer<float>;
template class Optimizer<double>;

}  // namespace cnn
