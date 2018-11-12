#pragma once

#include <memory>
#include <string>

#include "proto/cnn.pb.h"

#include "cnn/network.hpp"

namespace cnn
{

template<typename Dtype>
class Optimizer
{
 public:
    explicit Optimizer(const OptimizerProto& _proto);
    explicit Optimizer(const std::string& filename);
    void init(const OptimizerProto& _proto);

 private:
    OptimizerProto proto_;

    std::shared_ptr<Network<Dtype>> network_;
};

}  // namespace cnn
