#pragma once

#include <map>
#include <string>
#include <vector>

#include "proto/cnn.pb.h"

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

    const NetworkProto& proto() const {return proto_;}
    NetworkProto& proto() {return proto_;}

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

    std::shared_ptr<Layer<Dtype>> layer(int i) const
    {
        return layers_[i];
    }

    std::vector<const Array<Dtype>*> get_data_bottom(int i);
    std::vector<const Array<Dtype>*> get_data_top(int i);
    std::vector<Array<Dtype>*> get_data_top_mutable(int i);

    std::vector<Array<Dtype>*> get_gradient_bottom_mutable(int i);
    std::vector<const Array<Dtype>*> get_gradient_bottom(int i);
    std::vector<const Array<Dtype>*> get_gradient_top(int i);
    std::vector<Array<Dtype>*> get_gradient_top_mutable(int i);

 private:
    // add data to the map
    void add_data(const std::string& name,
            std::shared_ptr<Array<Dtype>> arr);

    void add_gradient(const std::string& name,
            std::shared_ptr<Array<Dtype>> arr);

 private:
    NetworkProto proto_;

    /** it saves the input and output of all layers in the network*/
    std::map<std::string, std::shared_ptr<Array<Dtype>>> data_;
    std::map<std::string, std::shared_ptr<Array<Dtype>>> gradient_;

    std::vector<std::shared_ptr<Layer<Dtype>>> layers_;
};

}  // namespace cnn

