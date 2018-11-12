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
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& /*top_gradient*/)
{
    CHECK_EQ(bottom.size(), 2)
        << "It should have two inputs where "
        << "input[0] is the predication and "
        << "input[1] is the ground truth";

    CHECK(bottom[0]->has_same_shape(*bottom[1]));

    CHECK_EQ(top.size(), 1);
    top[0]->init(1, 1, 1, 1);

    if (this->proto_.phase() == TRAIN)
    {
        CHECK_GE(bottom_gradient.size(), 1);

        bottom_gradient[0]->init_like(*(bottom[0]));
        // we do not use the bottom_gradient[1] which is for the label
    }
}

template<typename Dtype>
void L2LossLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
    loss_ =  ax_sub_by_squared<Dtype>(
            bottom[0]->total_,
            1, bottom[0]->d_,
            1, bottom[1]->d_);

    loss_ /= bottom[0]->total_;

    top[0]->d_[0] = loss_;
}

template<typename Dtype>
void L2LossLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& /*top_gradient*/)
{
    // top[0] is the loss
#if 0
    Dtype scale = top[0]->d_[0];
#else
    (void) top;
    Dtype scale = 1;
#endif
    // TODO(fangjun) it can be optimized, i.e., with lapack or blas.
    for (int i = 0; i < bottom[0]->total_; i++)
    {
        Dtype estimated_y = bottom[0]->d_[i];
        Dtype true_y = bottom[1]->d_[i];

        bottom_gradient[0]->d_[i] = scale * (estimated_y - true_y);
    }
}

template class L2LossLayer<float>;
template class L2LossLayer<double>;

}  // namespace cnn
