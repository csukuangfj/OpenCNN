#include <glog/logging.h>

#include <vector>

#include "cnn/convolution_layer.hpp"

namespace
{
bool is_inside(int h, int w, int height, int width)
{
    return (h >= 0) && (h < height)
        && (w >= 0) && (w < width);
}

}

namespace cnn
{
template<typename Dtype>
ConvolutionLayer<Dtype>::ConvolutionLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    const auto& p = _proto.conv_proto();
    num_output_ = p.num_output();
    kernel_size_ = p.kernel_size();

    CHECK_GE(num_output_, 1);
    CHECK_GE(kernel_size_, 1);
    CHECK(kernel_size_ & 1)
        << "the kernel size must be odd!";
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& top_gradient)
{
    CHECK_EQ(bottom.size(), 1);
    CHECK_EQ(top.size(), 1);

    top[0]->init(
            bottom[0]->n_,
            num_output_,
            bottom[0]->h_,
            bottom[0]->w_);

    if (this->param_.empty())
    {
        // param[0] is the kernel weight
        // param[1] is the bias
        this->param_.resize(2);

        this->param_[0] = std::make_shared<Array<Dtype>>();
        this->param_[0]->init(
                num_output_,
                bottom[0]->c_,
                kernel_size_,
                kernel_size_);

        // TODO(fangjun): use other strategies
        gaussian<Dtype>(this->param_[0].get(), 0, 1);

        this->param_[1] = std::make_shared<Array<Dtype>>();
        this->param_[1]->init(1, 1, 1, num_output_);

        // TODO(fangjun): use other strategies
        gaussian<Dtype>(this->param_[1].get(), 0, 1);
    }
    else    // NOLINT
    {
        CHECK_EQ(this->param_.size(), 2);

        CHECK_EQ(this->param_[0]->n_, num_output_);
        CHECK_EQ(this->param_[0]->c_, bottom[0]->c_);
        CHECK_EQ(this->param_[0]->h_, kernel_size_);
        CHECK_EQ(this->param_[0]->w_, kernel_size_);

        CHECK(this->param_[1]->has_same_shape({1, 1, 1, num_output_}));
    }

    if (this->proto().phase() == TRAIN)
    {
        // gradient for parameters
        this->gradient_.resize(2);
        this->gradient_[0].reset(new Array<Dtype>);
        this->gradient_[1].reset(new Array<Dtype>);

        this->gradient_[0]->init_like(*this->param_[0]);
        this->gradient_[1]->init_like(*this->param_[1]);

        // gradient for the bottom input
        CHECK_EQ(bottom_gradient.size(), 1);
        if (!bottom_gradient[0]->has_same_shape(*bottom[0]))
        {
            bottom_gradient[0]->init_like(*bottom[0]);
        }

        // gradient for the top input
        CHECK_EQ(top_gradient.size(), 1);
        top_gradient[0]->init_like(*top[0]);
    }
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
    const auto& b = *bottom[0];
    auto& t = *top[0];

    int h = b.h_;
    int w = b.w_;

    for (int n = 0; n < b.n_; n++)
    for (int i = 0; i < num_output_; i++)
    {
        for (int c = 0; c < b.c_; c++)
        {
            one_channel_convolution(
                    &this->param_[0]->operator()(i, c, 0, 0),
                    &b(n, c, 0, 0),
                    b.h_,
                    b.w_,
                    &t(n, i, 0, 0));
        }
        // add the bias
        sub_scalar<Dtype>(
                h*w,
                -this->param_[1]->d_[i],
                &t(n, i, 0, 0),
                &t(n, i, 0, 0));
    }
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& /*top*/,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
    const auto& b = *bottom[0];
    auto& bg = *bottom_gradient[0];

    const auto& tg = *top_gradient[0];

    for (int n = 0; n < b.n_; n++)
    for (int i = 0; i < num_output_; i++)
    {
        for (int c = 0; c < b.c_; c++)
        {
            one_channel_bprop(
                    &this->param_[0]->operator()(i, c, 0, 0),
                    &b(n, c, 0, 0),
                    b.h_,
                    b.w_,
                    &tg(n, i, 0, 0),
                    &bg(n, c, 0, 0),
                    &this->gradient_[0]->operator()(i, c, 0, 0));
        }
        // gradient for the bias
        this->gradient_[1]->d_[i] += sum_arr(b.h_*b.w_, &tg(n, i, 0, 0));
    }
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::one_channel_convolution(
        const Dtype* weight,
        const Dtype* src,
        int height, int width,
        Dtype* dst)
{
    // we assume the anchor is at the center of the kernel
    int s = kernel_size_ / 2;
    for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++)
    {
        Dtype t = 0;
        for (int i = -s; i <= s; i++)
        for (int j = -s; j <= s; j++)
        {
            if (!is_inside(h+i, w+j, height, width))
            {
                continue;
            }

            Dtype pixel = src[(h+i)*width + (w+j)];
            Dtype scale = weight[(i+s)*kernel_size_ + (j+s)];
            t += pixel*scale;
        }

        dst[h*width + w] += t;
    }
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::one_channel_bprop(
        const Dtype* weight,
        const Dtype* bottom,
        int height, int width,
        const Dtype* top_gradient,
        Dtype* bottom_gradient,
        Dtype* param_gradient)
{
    int s = kernel_size_ / 2;
    for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++)
    {
        Dtype tg = top_gradient[h*width+w];
        for (int i = -s; i <= s; i++)
        for (int j = -s; j <= s; j++)
        {
            if (!is_inside(h+i, w+j, height, width))
            {
                continue;
            }

            bottom_gradient[(h+i)*width + (w+j)] += tg * weight[(i+s)*kernel_size_ + (j+s)];
            param_gradient[(i+s)*kernel_size_ + (j+s)] += tg * bottom[(h+i)*width + (w+j)];
        }
    }
}

}  // namespace cnn


