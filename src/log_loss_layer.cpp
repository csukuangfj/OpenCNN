#include <glog/logging.h>

#include <vector>

#include "cnn/array_math.hpp"
#include "cnn/log_loss_layer.hpp"

namespace cnn
{
template<typename Dtype>
LogLossLayer<Dtype>::LogLossLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto),
      loss_(0)
{}

template<typename Dtype>
void LogLossLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& /*top_gradient*/)
{
    CHECK_EQ(bottom.size(), 2)
        << "It should have two inputs where "
        << "input[0] is the predication and "
        << "input[1] is the ground truth";

    CHECK_EQ(bottom[0]->n_, bottom[1]->n_);

    CHECK_EQ(bottom[1]->c_, 1);
    CHECK_EQ(bottom[0]->h_, bottom[1]->h_);
    CHECK_EQ(bottom[0]->w_, bottom[1]->w_);

    CHECK_EQ(top.size(), 1);
    top[0]->init(1, 1, 1, 1);

    if (this->proto_.phase() == TRAIN)
    {
        CHECK_GE(bottom_gradient.size(), 1);

        if (!bottom_gradient[0]->has_same_shape(*bottom[0]))
        {
            bottom_gradient[0]->init_like(*(bottom[0]));
        }
        // we do not use the bottom_gradient[1] which is for the label
    }
}

template<typename Dtype>
void LogLossLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
}

template<typename Dtype>
void LogLossLayer<Dtype>::bprop(
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
}

template class LogLossLayer<float>;
template class LogLossLayer<double>;

}  // namespace cnn

