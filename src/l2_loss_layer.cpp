#include <glog/logging.h>

#include <vector>

#include "cnn/array_math.hpp"
#include "cnn/l2_loss_layer.hpp"

namespace cnn
{
template<typename Dtype>
L2LossLayer<Dtype>::L2LossLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto),
      loss_(0)
{}

template<typename Dtype>
void L2LossLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    CHECK_EQ(input.size(), 2)
        << "It should have two inputs where "
        << "input[0] is the predication and "
        << "input[1] is the ground truth";

    CHECK_EQ(input[0]->n_, input[1]->n_);
    CHECK_EQ(input[0]->c_, input[1]->c_);
    CHECK_EQ(input[0]->h_, input[1]->h_);
    CHECK_EQ(input[0]->w_, input[1]->w_);

    CHECK_EQ(output.size(), 1);
    output[0]->init(1, 1, 1, 1);
}

template<typename Dtype>
void L2LossLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    loss_ = output[0]->d_[0] = ax_sub_by_squared<Dtype>(
            input[0]->total_,
            1, input[0]->d_,
            1, input[1]->d_);
}

template<typename Dtype>
void L2LossLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

template class L2LossLayer<float>;
template class L2LossLayer<double>;

}  // namespace cnn
