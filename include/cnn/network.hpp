#pragma once

#include "proto/cnn.pb.h"

namespace cnn
{
template<typename Dtype>
class Network
{
 public:
    explicit Network(const NetworkProto&);
    explicit Network(const std::string &filename, bool is_binary = false);
    void init(const std::string &filename, bool is_binary = false);
    void init(const NetworkProto&);

 private:
    NetworkProto proto_;
};

}  // namespace cnn

