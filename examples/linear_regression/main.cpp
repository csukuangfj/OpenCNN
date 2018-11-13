#include <glog/logging.h>

#include <sstream>
#include <string>
#include <vector>

#include "cnn/optimizer.hpp"

template<typename Dtype>
void data_callback(const std::vector<cnn::Array<Dtype>*> &top)
{
    // y = 5 + 10*x
    static std::vector<std::pair<std::vector<Dtype>, Dtype>> data{
        {{11}, 115},
        {{-14}, -135},
        {{15}, 155},
        {{6}, 65},
        {{-18}, -175},
        {{-8}, -75},
        {{9}, 95},
        {{-4}, -35},
        {{18}, 185},
        {{-1}, -5},
    };

    static int k = 0;

    int n = top[0]->n_;

    CHECK_LE(n, data.size())
        << "the batch size cannot be larger than the dataset size";

    int stride = top[0]->total_/n;
    CHECK_EQ(stride, data[0].first.size());

    for (int i = 0; i < n; i++)
    {
        if (k >= data.size())
        {
            k = 0;
        }

        for (int j = 0; j < stride; j++)
        {
            top[0]->d_[i*stride + j] = (data[k].first)[j];
        }

        if (top.size() == 2)
        {
            top[1]->d_[i] = data[k].second;
        }

        k++;
    }
}

int main(int /*argc*/, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    std::string filename = "../examples/linear_regression/optimizer.prototxt";

    cnn::Optimizer<double> opt(filename);
    opt.register_data_callback(data_callback);
    opt.start_training();

    return 0;
}

