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
