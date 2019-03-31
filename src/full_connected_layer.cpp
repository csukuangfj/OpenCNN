#include <glog/logging.h>

#include <vector>

#include "cnn/array_math.hpp"
#include "cnn/full_connected_layer.hpp"
#include "cnn/jet.hpp"
#include "cnn/rng.hpp"

namespace cnn {
template <typename Dtype>
FullConnectedLayer<Dtype>::FullConnectedLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto) {
  num_output_ = _proto.fc_proto().num_output();
}

template <typename Dtype>
void FullConnectedLayer<Dtype>::reshape(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<Array<Dtype>*>& top,
    const std::vector<Array<Dtype>*>& top_gradient) {
  CHECK_EQ(bottom.size(), 1);

  CHECK_EQ(top.size(), 1);

  int n = bottom[0]->n_;
  top[0]->init(n, num_output_, 1, 1);

  if (this->param_.empty()) {
    this->param_.resize(2);
    this->param_[0] = std::make_shared<Array<Dtype>>();
    this->param_[0]->init(1, 1, num_output_, bottom[0]->total_ / n);

    // TODO(fangjun): use other strategies
    gaussian<Dtype>(this->param_[0].get(), 0, 1);

    this->param_[1] = std::make_shared<Array<Dtype>>();
    this->param_[1]->init(1, 1, 1, num_output_);

    // TODO(fangjun): use other strategies
    gaussian<Dtype>(this->param_[1].get(), 0, 1);
  } else  // NOLINT
  {
    CHECK_EQ(this->param_.size(), 2);

    CHECK_EQ(this->param_[0]->n_, 1);
    CHECK_EQ(this->param_[0]->c_, 1);
    CHECK_EQ(this->param_[0]->h_, num_output_);
    CHECK_EQ(this->param_[0]->w_, bottom[0]->total_ / n);

    CHECK(this->param_[1]->has_same_shape({1, 1, 1, num_output_}));
  }

  if (this->proto().phase() == TRAIN) {
    // gradient for parameters
    this->gradient_.resize(2);
    this->gradient_[0].reset(new Array<Dtype>);
    this->gradient_[1].reset(new Array<Dtype>);

    this->gradient_[0]->init_like(*this->param_[0]);
    this->gradient_[1]->init_like(*this->param_[1]);

    // gradient for the bottom input
    CHECK_EQ(bottom_gradient.size(), 1);
    if (!bottom_gradient[0]->has_same_shape(*bottom[0])) {
      bottom_gradient[0]->init_like(*bottom[0]);
    }

    // gradient for the top input
    CHECK_EQ(top_gradient.size(), 1);
    top_gradient[0]->init_like(*top[0]);
  }
}

template <typename Dtype>
void FullConnectedLayer<Dtype>::fprop(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& top) {
  int n = bottom[0]->n_;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < num_output_; j++) {
      Dtype dot = ax_dot_by<Dtype>(this->param_[0]->w_, 1,
                                   &this->param_[0]->operator()(0, 0, j, 0), 1,
                                   &bottom[0]->operator()(i, 0, 0, 0));
      top[0]->operator()(i, j, 0, 0) = dot + this->param_[1]->operator[](j);
    }
  }
}

template <typename Dtype>
void FullConnectedLayer<Dtype>::bprop(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<const Array<Dtype>*>& top,
    const std::vector<const Array<Dtype>*>& top_gradient) {
  // compute parameter gradient
  auto& w = *this->param_[0];
  auto& dw = *this->gradient_[0];

  auto& db = *this->gradient_[1];

  auto& x = *bottom[0];
  auto& dx = *bottom_gradient[0];

  auto& y = *top[0];
  auto& dy = *top_gradient[0];

  int stride = dw.w_;
  for (int n = 0; n < y.n_; n++) {
    for (int i = 0; i < num_output_; i++) {
      Dtype scale = dy(n, i, 0, 0);
      ax_plus_by<Dtype>(stride, scale, &x[n * stride], 1, &dw(0, 0, i, 0));

      db.d_[i] += scale;

      ax_plus_by<Dtype>(stride, scale, &w(0, 0, i, 0), 1, &dx[n * stride]);
    }
  }
}

}  // namespace cnn
