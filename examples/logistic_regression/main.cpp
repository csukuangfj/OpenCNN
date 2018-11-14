#include <glog/logging.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cnn/optimizer.hpp"

namespace
{
std::vector<std::pair<std::vector<double>, double>> g_data;

void load_data(const std::string& filename)
{
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open())
    {
        LOG(FATAL) << "cannot open " << filename;
    }

    // we omit sanity check in the following
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        double s1, s2, label;

        ss >> s1; ss.ignore();
        ss >> s2; ss.ignore();
        ss >> label; ss.ignore();

        g_data.push_back({{s1, s2}, label});
    }
}

template<typename Dtype>
void data_callback(const std::vector<cnn::Array<Dtype>*> &top)
{
    static int k = 0;
    int n = top[0]->n_;
    CHECK_LE(n, g_data.size())
        << "the batch size cannot be larger than the dataset size";

    int stride = top[0]->total_/n;
    CHECK_EQ(stride, g_data[0].first.size());

    for (int i = 0; i < n; i++)
    {
        if (k >= g_data.size())
        {
            k = 0;
        }

        for (int j = 0; j < stride; j++)
        {
            top[0]->d_[i*stride + j] = (g_data[k].first)[j];
        }

        if (top.size() == 2)
        {
            top[1]->d_[i] = g_data[k].second;
        }

        k++;
    }
}


}


int main(int /*argc*/, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    load_data("../examples/logistic_regression/data/ex2data1.txt");
    CHECK_EQ(g_data.size(), 100)
        << "the file should contain 100 lines!";

    std::string filename = "../examples/logistic_regression/trained.prototxt";
    cnn::Network<double> network(filename);
    network.reshape();

    std::ostringstream ss;
    ss << "\n";
    int correct = 0;
    int total = 0;

    network.get_data_top_mutable(0)[0]->d_[0] = 45;
    network.get_data_top_mutable(0)[0]->d_[1] = 85;

    network.perform_predication();
    auto predictions = network.get_predications();

    LOG(INFO) << "0: " << predictions[0];
    LOG(INFO) << "1: " << predictions[1];

#if 1
    for (int i = 0; i < g_data.size(); i++)
    // for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < g_data[i].first.size(); j++)
        {
            network.get_data_top_mutable(0)[0]->d_[j] = g_data[i].first[j];
        }
        network.perform_predication();

        auto predictions = network.get_predications();
        int res = predictions[1] >= predictions[0];

        ss << "actual: " << res << "\n";
        ss << "expected: " << g_data[i].second;
        total++;
        correct += res == int(g_data[i].second);

        ss << "\n\n";
    }

    LOG(INFO) << ss.str();
    LOG(INFO) << "accuracy: " << (float)correct/total;

    network.get_data_top_mutable(0)[0]->d_[0] = 45;
    network.get_data_top_mutable(0)[0]->d_[1] = 85;

    network.perform_predication();
    predictions = network.get_predications();

    LOG(INFO) << "0: " << predictions[0];
    LOG(INFO) << "1: " << predictions[1];
#else
    std::string filename = "../examples/logistic_regression/optimizer.prototxt";

    cnn::Optimizer<double> opt(filename);
    opt.register_data_callback(data_callback);
    opt.start_training();
#endif

    return 0;
}

