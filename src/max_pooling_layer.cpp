#include <glog/logging.h>

#include <vector>

#include "cnn/max_pooling_layer.hpp"

namespace cnn
{

template<typename Dtype>
MaxPoolingLayer<Dtype>::MaxPoolingLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    const auto& p = _proto.max_pooling_proto();
    win_size_ = p.win_size();
    stride_ = p.stride();

    CHECK_GT(win_size_, 1) << "window size must be greater than 1";

    CHECK_GT(stride_, 0) << "stride size must be greater than 0";
}

template<typename Dtype>
void MaxPoolingLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& top_gradient)
{
    CHECK_EQ(bottom.size(), 1) << "max pooling accepts only 1 input";
    CHECK_EQ(top.size(), 1) << "max pooling generates only 1 output";

    int h = (bottom[0]->h_ - win_size_) / stride_ + 1;
    int w = (bottom[0]->w_ - win_size_) / stride_ + 1;

    top[0]->init(
            bottom[0]->n_,
            bottom[0]->c_,
            h,
            w);

    if (this->proto_.phase() == TRAIN)
    {
        CHECK_EQ(bottom_gradient.size(), 1);

        if (!bottom_gradient[0]->has_same_shape(*bottom[0]))
        {
            bottom_gradient[0]->init_like(*bottom[0]);
        }

        CHECK_EQ(top_gradient.size(), 1);
        top_gradient[0]->init_like(*top[0]);
    }
}

template<typename Dtype>
void MaxPoolingLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
}

template<typename Dtype>
void MaxPoolingLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& /*top*/,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

}  // namespace cnn



