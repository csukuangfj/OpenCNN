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

#include "cnn/network.hpp"

int main(int /*argc*/, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::string filename = "../proto/trained.prototxt";
  cnn::Network<double> network(filename);
  network.reshape();

  // y = 5 + 10x
  std::ostringstream ss;
  ss << "\n";
  for (int i = 0; i < 10; i++) {
    network.get_data_top_mutable(0)[0]->d_[0] = i;
    network.perform_predication();

    ss << "expect: " << (5 + 10 * i) << ", ";
    ss << "actual: " << network.get_predications()[0];
    ss << "\n";
  }

  LOG(INFO) << ss.str();

  return 0;
}
