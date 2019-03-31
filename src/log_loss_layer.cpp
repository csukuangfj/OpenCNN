
#include <algorithm>
#include <vector>

#include "cnn/array_math.hpp"
#include "cnn/common.hpp"
#include "cnn/log_loss_layer.hpp"

namespace cnn {

template <typename Dtype>
LogLossLayer<Dtype>::LogLossLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto), loss_(0) {}

template <typename Dtype>
void LogLossLayer<Dtype>::reshape(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<Array<Dtype>*>& top,
    const std::vector<Array<Dtype>*>& /*top_gradient*/) {
  CHECK_EQ(bottom.size(), 2) << "It should have two inputs where "
                             << "input[0] is the predication and "
                             << "input[1] is the ground truth";

  CHECK_EQ(bottom[0]->n_, bottom[1]->n_);

  CHECK_EQ(bottom[1]->c_, 1);
  CHECK_EQ(bottom[0]->h_, bottom[1]->h_);
  CHECK_EQ(bottom[0]->w_, bottom[1]->w_);

  CHECK_EQ(top.size(), 1);
  top[0]->init(1, 1, 1, 1);

  if (this->proto_.phase() == TRAIN) {
    CHECK_GE(bottom_gradient.size(), 1);

    if (!bottom_gradient[0]->has_same_shape(*bottom[0])) {
      bottom_gradient[0]->init_like(*(bottom[0]));
    }
    // we do not use the bottom_gradient[1] which is for the label
  }
}

template <typename Dtype>
void LogLossLayer<Dtype>::fprop(const std::vector<const Array<Dtype>*>& bottom,
                                const std::vector<Array<Dtype>*>& top) {
  loss_ = 0;

  const auto& b0 = *bottom[0];
  const auto& b1 = *bottom[1];

  for (int n = 0; n < b1.n_; n++)
    for (int h = 0; h < b1.h_; h++)
      for (int w = 0; w < b1.w_; w++) {
        auto label = b1(n, 0, h, w);  // label for the ground truth
        auto p = b0(n, label, h, w);  // probability for the predication
        p = std::max(p, Dtype(g_log_threshold));
        p = std::min(p, Dtype(1));

        // use cnn::log() here for Jet, in unit test only.
        loss_ += Dtype(-1) * log(p);
      }

  loss_ /= b1.total_;  // take the average

  top[0]->d_[0] = loss_;
}

template <typename Dtype>
void LogLossLayer<Dtype>::bprop(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<const Array<Dtype>*>& /*top*/,
    const std::vector<const Array<Dtype>*>& /*top_gradient*/) {
  Dtype scale = 1;

  const auto& b0 = *bottom[0];
  const auto& b1 = *bottom[1];
  auto& bg = *bottom_gradient[0];

  scale /= Dtype(-1) * b1.total_;
  for (int n = 0; n < b1.n_; n++)
    for (int h = 0; h < b1.h_; h++)
      for (int w = 0; w < b1.w_; w++) {
        auto label = b1(n, 0, h, w);

        auto p = b0(n, label, h, w);  // probability for the predication
        p = std::max(p, Dtype(g_log_threshold));
        p = std::min(p, Dtype(1));
        bg(n, label, h, w) = scale / p;
      }
}

}  // namespace cnn
