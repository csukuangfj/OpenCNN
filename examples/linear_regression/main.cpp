/*  ---------------------------------------------------------------------
  Copyright 2018-2019 Fangjun Kuang
  email: csukuangfj at gmail dot com
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a COPYING file of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>
  -----------------------------------------------------------------  */
#include <glog/logging.h>

#include <sstream>
#include <string>
#include <vector>

#include "cnn/optimizer.hpp"

template <typename Dtype>
void data_callback(const std::vector<cnn::Array<Dtype>*>& top) {
  // y = 5 + 10*x
  static std::vector<std::pair<std::vector<Dtype>, Dtype>> data{
      {{11}, 115}, {{-14}, -135}, {{15}, 155}, {{6}, 65},   {{-18}, -175},
      {{-8}, -75}, {{9}, 95},     {{-4}, -35}, {{18}, 185}, {{-1}, -5},
  };

  static int k = 0;

  int n = top[0]->n_;

  CHECK_LE(n, data.size())
      << "the batch size cannot be larger than the dataset size";

  int stride = top[0]->total_ / n;
  CHECK_EQ(stride, data[0].first.size());

  for (int i = 0; i < n; i++) {
    if (k >= data.size()) {
      k = 0;
    }

    for (int j = 0; j < stride; j++) {
      top[0]->d_[i * stride + j] = (data[k].first)[j];
    }

    if (top.size() == 2) {
      top[1]->d_[i] = data[k].second;
    }

    k++;
  }
}

int main(int /*argc*/, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::string filename = "../examples/linear_regression/optimizer.prototxt";

  cnn::Optimizer<double> opt(filename);
  opt.register_data_callback(data_callback);
  opt.start_training();

  return 0;
}
