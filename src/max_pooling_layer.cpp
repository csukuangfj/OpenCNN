#include <glog/logging.h>

#include <utility>
#include <vector>

#include "cnn/max_pooling_layer.hpp"

namespace cnn {

template <typename Dtype>
MaxPoolingLayer<Dtype>::MaxPoolingLayer(const LayerProto& _proto)
    : Layer<Dtype>(_proto) {
  const auto& p = _proto.max_pooling_proto();
  win_size_ = p.win_size();
  stride_ = p.stride();

  CHECK_GT(win_size_, 1) << "window size must be greater than 1";

  CHECK_GT(stride_, 0) << "stride size must be greater than 0";
}

template <typename Dtype>
void MaxPoolingLayer<Dtype>::reshape(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<Array<Dtype>*>& top,
    const std::vector<Array<Dtype>*>& top_gradient) {
  CHECK_EQ(bottom.size(), 1) << "max pooling accepts only 1 input";
  CHECK_EQ(top.size(), 1) << "max pooling generates only 1 output";

  int h = (bottom[0]->h_ - win_size_) / stride_ + 1;
  int w = (bottom[0]->w_ - win_size_) / stride_ + 1;

  top[0]->init(bottom[0]->n_, bottom[0]->c_, h, w);

  max_index_pair_.init_like(*top[0]);

  if (this->proto_.phase() == TRAIN) {
    CHECK_EQ(bottom_gradient.size(), 1);

    if (!bottom_gradient[0]->has_same_shape(*bottom[0])) {
      bottom_gradient[0]->init_like(*bottom[0]);
    }

    CHECK_EQ(top_gradient.size(), 1);
    top_gradient[0]->init_like(*top[0]);
  }
}

template <typename Dtype>
void MaxPoolingLayer<Dtype>::fprop(
    const std::vector<const Array<Dtype>*>& bottom,
    const std::vector<Array<Dtype>*>& top) {
  const auto& b = *bottom[0];
  auto& t = *top[0];
  for (int n = 0; n < t.n_; n++)
    for (int c = 0; c < t.c_; c++)
      for (int h = 0; h < t.h_; h++)
        for (int w = 0; w < t.w_; w++) {
          auto p =
              find_max_index(&b(n, c, 0, 0), b.w_, h * stride_, w * stride_);

          t(n, c, h, w) = b(n, c, p.first, p.second);
          max_index_pair_(n, c, h, w) = p;
        }
}

template <typename Dtype>
void MaxPoolingLayer<Dtype>::bprop(
    const std::vector<const Array<Dtype>*>& /*bottom*/,
    const std::vector<Array<Dtype>*>& bottom_gradient,
    const std::vector<const Array<Dtype>*>& /*top*/,
    const std::vector<const Array<Dtype>*>& top_gradient) {
  auto& bg = *bottom_gradient[0];
  const auto& tg = *top_gradient[0];
  for (int n = 0; n < tg.n_; n++)
    for (int c = 0; c < tg.c_; c++)
      for (int h = 0; h < tg.h_; h++)
        for (int w = 0; w < tg.w_; w++) {
          const auto& p = max_index_pair_(n, c, h, w);
          bg(n, c, p.first, p.second) += tg(n, c, h, w);
        }
}

template <typename Dtype>
std::pair<int, int> MaxPoolingLayer<Dtype>::find_max_index(const Dtype* arr,
                                                           int width, int h,
                                                           int w) const {
  // find the index of the max value in the window
  // [h, h+win_size_) x [w, w+win_size_)
  Dtype max_val = arr[h * width + w];
  int max_h = h;
  int max_w = w;

  for (int i = h; i < h + win_size_; i++)
    for (int j = w; j < w + win_size_; j++) {
      const auto& val = arr[i * width + j];
      if (val > max_val) {
        max_val = val;
        max_h = i;
        max_w = j;
      }
    }
  return {max_h, max_w};
}

}  // namespace cnn
