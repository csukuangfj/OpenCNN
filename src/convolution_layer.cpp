#include <glog/logging.h>

#include <vector>

#include "cnn/convolution_layer.hpp"

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
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

}  // namespace cnn


