#include <glog/logging.h>

#include <vector>

#include "cnn/array_math.hpp"
#include "cnn/full_connected_layer.hpp"

namespace cnn
{
template<typename Dtype>
FullConnectedLayer<Dtype>::FullConnectedLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    num_output_ = _proto.fc_proto().num_output();
}


template<typename Dtype>
void FullConnectedLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    CHECK_EQ(input.size(), 1);

    // resize output
    CHECK_EQ(output.size(), 1);
    int n = input[0]->n_;
    int c = num_output_;
    int h = 1;
    int w = 1;
    output[0]->init(n, c, h, w);

    if (this->param_.empty())
    {
        // resize param_
        this->param_.resize(2);
        this->param_[0] = std::make_shared<Array<Dtype>>();
        this->param_[0]->init(1, 1, num_output_, input[0]->total_/n);
        gaussian<Dtype>(this->param_[0].get(), 0, 1);

        this->param_[1] = std::make_shared<Array<Dtype>>();
        this->param_[1]->init(1, 1, 1, num_output_);
        gaussian<Dtype>(this->param_[1].get(), 0, 1);
    }
    else    // NOLINT
    {
        CHECK_EQ(this->param_.size(), 2);
        CHECK_EQ(this->param_[0]->h_, num_output_);
        CHECK_EQ(this->param_[0]->w_, input[0]->total_/n);

        CHECK_EQ(this->param_[1]->total_, num_output_);
    }
}

template<typename Dtype>
void FullConnectedLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& input,
        const std::vector<Array<Dtype>*>& output)
{
    int n = input[0]->n_;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < num_output_; j++)
        {
            Dtype dot = ax_dot_by<Dtype>(this->param_[0]->w_,
                    1,
                    &this->param_[0]->operator()(0, 0, j, 0),
                    1,
                    &input[0]->operator()(i, 0, 0, 0));
            output[0]->operator()(i, j, 0, 0) =
                dot + this->param_[1]->operator[](j);
        }
    }
}

template<typename Dtype>
void FullConnectedLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<const Array<Dtype>*>& top,
        const std::vector<const Array<Dtype>*>& top_gradient)
{
}

template class FullConnectedLayer<float>;
template class FullConnectedLayer<double>;

}  // namespace cnn

