#pragma once

#include "proto/cnn.pb.h"

#include <map>
#include <string>
#include <vector>

#include "cnn/array.hpp"
#include "cnn/layer.hpp"

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

    void reshape();
    /** forward propagation */
    void fprop();

    /** backward propagation */
    void bprop();

    /** compute the loss for the last forward propagation.
     * No forward propagation is performed here; it just gets
     * the loss from the last loss layer.
     */
    Dtype compute_loss();
 private:
    // add data to the map
    void add_data(const std::string& name,
            std::shared_ptr<Array<Dtype>> arr);

    std::vector<const Array<Dtype>*> get_data_input(int i);
    std::vector<Array<Dtype>*> get_data_output(int i);

 private:
    NetworkProto proto_;

    /** it saves the input and output of all layers in the network*/
    std::map<std::string, std::shared_ptr<Array<Dtype>>> data_;

    std::vector<std::shared_ptr<Layer<Dtype>>> layers_;
};

}  // namespace cnn

