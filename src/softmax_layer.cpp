#include <glog/logging.h>

#include <limits>
#include <vector>

#include "cnn/softmax_layer.hpp"

namespace cnn
{

template<typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{}

template<typename Dtype>
void SoftmaxLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& top_gradient)
{
    CHECK_EQ(bottom.size(), 1) << "softmax accepts only 1 input";
    CHECK_EQ(top.size(), 1) << "softmax generates only 1 output";

    top[0]->init_like(*bottom[0]);

    if (this->proto_.phase() == TRAIN)
    {
        CHECK_EQ(bottom_gradient.size(), 1);

        if (!bottom_gradient[0]->has_same_shape(*bottom[0]))
        {
            bottom_gradient[0]->init_like(*bottom[0]);
        }

        top_gradient[0]->init_like(*bottom[0]);
    }

    CHECK_GE(bottom[0]->c_, 2)
        << "we need at least two value to compute softmax!";

    buffer_.init(1, 1, 1, bottom[0]->c_);
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
    const auto& d = *bottom[0];
    auto& target = *top[0];
    for (int n = 0; n < d.n_; n++)
    for (int h = 0; h < d.h_; h++)
    for (int w = 0; w < d.w_; w++)
    {
        Dtype max_val = std::numeric_limits<Dtype>::min();
        for (int c = 0; c < d.c_; c++)
        {
            max_val = (d(n, c, h, w) > max_val) ? buffer_[c] : max_val;
        }

        for (int c = 0; c < d.c_; c++)
        {
            buffer_[c] = std::exp(d(n, c, h, w) - max_val);
        }

        Dtype den = sum_arr(buffer_);
        Dtype scale = Dtype(1) / den;
        scale_arr(scale, buffer_, &buffer_);
        for (int c = 0; c < d.c_; c++)
        {
            target(n, c, h, w) = buffer_[c];
        }
    }
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
    auto& bg = *bottom_gradient[0];
    const auto& tg = *top_gradient[0];
    const auto& t = *top[0];

    for (int n = 0; n < bottom[0]->n_; n++)
    for (int h = 0; h < bottom[0]->h_; h++)
    for (int w = 0; w < bottom[0]->w_; w++)
    for (int c = 0; c < bottom[0]->c_; c++)
    {
        Dtype yc = t(n, c, h, w);
        // TODO(fangjun) optimize it, i.e., eliminate the inner else condition
        // TODO(fangjun) implement copy_arr() to copy an array
        for (int i = 0; i < bottom[0]->c_; i++)
        {
            Dtype scale = tg(n, i, h, w);
            Dtype yi = t(n, i, h, w);
            if (i == c)
            {
                bg(n, c, h, w) += scale * (yi - yi*yc);
            }
            else    // NOLINT
            {
                bg(n, c, h, w) += scale * (-yi * yc);
            }
        }
    }
}

template class SoftmaxLayer<float>;
template class SoftmaxLayer<double>;

}  // namespace cnn

