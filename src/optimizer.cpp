
#include <glog/logging.h>

#include <cmath>    // std::pow
#include <fstream>  // NOLINT
#include <string>

#include "cnn/array_math.hpp"
#include "cnn/io.hpp"
#include "cnn/optimizer.hpp"

namespace cnn {
template <typename Dtype>
Optimizer<Dtype>::Optimizer(const OptimizerProto& _proto) {
  init(_proto);
}

template <typename Dtype>
Optimizer<Dtype>::Optimizer(const std::string& filename) {
  OptimizerProto _proto;
  read_proto_txt(filename, &_proto);
  init(_proto);
}

template <typename Dtype>
void Optimizer<Dtype>::init(const OptimizerProto& _proto) {
  proto_ = _proto;

  auto network_filename = proto_.model_filename();
  network_.reset(new Network<Dtype>(network_filename));
  if (proto_.has_trained_filename()) {
    network_->copy_trained_network(proto_.trained_filename(), true);
  }
}

template <typename Dtype>
void Optimizer<Dtype>::start_training() {
  std::ofstream of("loss.txt");
  int max_iter = proto_.max_iteration_num();
  network_->reshape();
  for (int i = 0; i < max_iter; i++) {
    network_->fprop();
    network_->bprop();
    update_parameters(i);

    if (i && !(i % proto_.print_interval())) {
      LOG(INFO) << "iter: " << i << ","
                << "loss is: " << network_->get_loss();
    }
    of << i << "," << network_->get_loss() << "\n";

    if (i && !(i % proto_.snapshot_interval())) {
      auto filename = proto_.snapshot_prefix() + "-" + std::to_string(i);
      network_->save_network(filename, true);
    }
  }

  LOG(INFO) << "iteration: " << max_iter;
  LOG(INFO) << "loss is: " << network_->get_loss();
  print_parameters();
  network_->save_network("trained-bin.prototxt", true);
}

template <typename Dtype>
void Optimizer<Dtype>::update_parameters(int current_iter) {
  auto& layers = network_->layers();
  int num_layers = layers.size();

  Dtype learning_rate = proto_.learning_rate();

  // TODO(fangjun): move the following options to proto
  static const double gamma = 0.0001;
  double base = 1 + gamma * current_iter;
  double exp = -0.75;
  learning_rate *= std::pow(base, exp);

  for (int i = 1; i < num_layers; i++) {
    layers[i]->update_parameters(current_iter, learning_rate);
  }
}

template <typename Dtype>
void Optimizer<Dtype>::print_parameters() {
  auto& layers = network_->layers();
  int num_layers = layers.size();

  std::ostringstream ss;
  ss << "\n";
  ss << "batch size is: " << network_->get_batch_size() << "\n";
  // we skip the input layer since it has no parameters
  for (int i = 1; i < num_layers; i++) {
#if 0
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
#endif
  }

  LOG(INFO) << ss.str();
}

}  // namespace cnn
