#include <glog/logging.h>

#include <vector>

#include "cnn/batch_normalization_layer.hpp"
#include "cnn/jet.hpp"
#include "cnn/rng.hpp"

namespace cnn
{

template<typename Dtype>
BatchNormalizationLayer<Dtype>::BatchNormalizationLayer(
        const LayerProto& _proto)
    : Layer<Dtype>(_proto)
{
    momentum_ = _proto.batch_normalization_proto().momentum();
    CHECK_GT(momentum_, 0);
    CHECK_LT(momentum_, 1);
}

template<typename Dtype>
void BatchNormalizationLayer<Dtype>::reshape(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& bottom_gradient,
        const std::vector<Array<Dtype>*>& top,
        const std::vector<Array<Dtype>*>& top_gradient)
{
    CHECK_EQ(bottom.size(), 1) << "Batch normalization accepts only 1 input";
    CHECK_EQ(top.size(), 1) << "Batch normalization generates only 1 output";

    top[0]->init_like(*bottom[0]);

    if (this->param_.empty())
    {
        this->param_.resize(4);

        // 0: scale
        // 1: bias
        // 2: mean
        // 3: stddev
        for (int i = 0; i < 4; i++)
        {
            this->param_[i].reset(new Array<Dtype>);
            this->param_[i]->init(1, bottom[0]->c_, 1, 1);
        }

        set_to<Dtype>(this->param_[0].get(), 1);   // default to no scale
        set_to<Dtype>(this->param_[1].get(), 0);   // default to no bias
        set_to<Dtype>(this->param_[2].get(), 0);   // 0 mean
        set_to<Dtype>(this->param_[3].get(), 0);   // 0 stddev
    }
    else    // NOLINT
    {
        CHECK_EQ(this->param_.size(), 4);
        for (int i = 0; i < 4; i++)
        {
            CHECK(this->param_[i]->has_same_shape({1, bottom[0]->c_, 1, 1}));
        }
    }

    if (this->proto_.phase() == TRAIN)
    {
        CHECK_EQ(bottom_gradient.size(), 1);

        if (!bottom_gradient[0]->has_same_shape(*bottom[0]))
        {
            bottom_gradient[0]->init_like(*bottom[0]);
        }

        CHECK_EQ(top_gradient.size(), 1);
        top_gradient[0]->init_like(*top[0]);

        // gradient for parameters: scale and bias
        this->gradient_.resize(2);
        this->gradient_[0].reset(new Array<Dtype>);
        this->gradient_[1].reset(new Array<Dtype>);

        this->gradient_[0]->init_like(*this->param_[0]);    // channel scale
        this->gradient_[1]->init_like(*this->param_[1]);    // channel bias
    }
}

template<typename Dtype>
void BatchNormalizationLayer<Dtype>::fprop(
        const std::vector<const Array<Dtype>*>& bottom,
        const std::vector<Array<Dtype>*>& top)
{
    const auto& b = *bottom[0];
    auto& t = *top[0];

    auto num_elements = b.h_ * b.w_;
    if (this->proto_.phase() == TRAIN)
    {
        Dtype num_batch_elements = num_elements * b.n_;
        for (int c = 0; c < b.c_; c++)
        {
            Dtype total = 0;

            // compute the sum across batches for the same channel
            for (int n = 0; n < b.n_; n++)
            {
                total += sum_arr(num_elements, &b(n, c, 0, 0));
            }

            // compute the mean
            Dtype mean = total / num_batch_elements;

            auto& moving_mean = this->param_[2]->d_[c];
            if (moving_mean == 0)
            {
                moving_mean = mean;
            }
            else    // NOLINT
            {
                moving_mean = moving_mean * momentum_
                    + mean * (Dtype(1) -  momentum_);
            }

            // subtract the mean
            for (int n = 0; n < b.n_; n++)
            {
                sub_scalar(num_elements, mean, &b(n, c, 0, 0), &t(n, c, 0, 0));
            }

            // compute the sum of square (x - mu)*(x - mu)
            total = 0;
            for (int n = 0; n < b.n_; n++)
            {
                total += sum_squared_arr(num_elements, &t(n, c, 0, 0));
            }

            total /= num_batch_elements;

            Dtype var = total + eps_;
            Dtype stddev = sqrt(var);

            auto& moving_stddev = this->param_[3]->d_[c];
            if (moving_stddev == 0)
            {
                moving_stddev = stddev;
            }
            else    // NOLINT
            {
                moving_stddev = moving_stddev * momentum_
                    + stddev * (Dtype(1) -  momentum_);
            }

            auto scale = this->param_[0]->d_[c] / stddev;
            auto bias = this->param_[1]->d_[c];

            for (int n = 0; n < b.n_; n++)
            {
                scale_arr(num_elements,
                        scale,
                        &t(n, c, 0, 0),
                        &t(n, c, 0, 0));

                sub_scalar(num_elements,
                        -bias,
                        &t(n, c, 0, 0),
                        &t(n, c, 0, 0));
            }
        }
    }   // if (... == TRAIN)
    else    // NOLINT
    {
        // TEST
        for (int c = 0; c < b.c_; c++)
        {
            Dtype mean = this->param_[2]->d_[c];

            // subtract the mean
            for (int n = 0; n < b.n_; n++)
            {
                sub_scalar(num_elements, mean, &b(n, c, 0, 0), &t(n, c, 0, 0));
            }

            // we have already added eps before
            Dtype stddev = this->param_[3]->d_[c];

            auto scale = this->param_[0]->d_[c] / stddev;
            auto bias = this->param_[1]->d_[c];

            for (int n = 0; n < b.n_; n++)
            {
                scale_arr(num_elements,
                        scale,
                        &t(n, c, 0, 0),
                        &t(n, c, 0, 0));

                sub_scalar(num_elements,
                        -bias,
                        &t(n, c, 0, 0),
                        &t(n, c, 0, 0));
            }
        }
    }
}

template<typename Dtype>
void BatchNormalizationLayer<Dtype>::bprop(
        const std::vector<const Array<Dtype>*>& /*bottom*/,
        const std::vector<Array<Dtype>*>& /*bottom_gradient*/,
        const std::vector<const Array<Dtype>*>& /*top*/,
        const std::vector<const Array<Dtype>*>& /*top_gradient*/)
{
}

}  // namespace cnn


