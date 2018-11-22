#include <glog/logging.h>

#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand

#include <string>
#include <fstream>
#include <vector>

#include "cnn/array.hpp"
#include "cnn/io.hpp"
#include "cnn/optimizer.hpp"

namespace
{

std::string g_path = "../examples/mnist/data";
std::string g_train_image_name = "train-images-idx3-ubyte";
std::string g_train_label_name = "train-labels-idx1-ubyte";

std::string g_test_image_name = "t10k-images-idx3-ubyte";
std::string g_test_label_name = "t10k-labels-idx1-ubyte";

std::vector<cnn::Array<uint8_t>> g_train_images;
std::vector<uint8_t> g_train_labels;

std::vector<cnn::Array<uint8_t>> g_test_images;
std::vector<uint8_t> g_test_labels;

std::vector<int> g_data_index;

int swap_endian(int in)
{
    int res = in;
    char* c = (char*)&res;

    std::swap(c[0], c[3]);
    std::swap(c[1], c[2]);
    return res;
}

std::vector<uint8_t> load_labels(const std::string& _filename)
{
    auto filename = g_path + "/" + _filename;
    FILE* f = fopen(filename.c_str(), "r");
    if (!f)
    {
        LOG(FATAL) << "failed to open " << filename;
    }
#define READ_INT32(x)                               \
    int x;                                          \
    CHECK_EQ(fread(&x, sizeof(int32_t), 1, f), 1);  \
    x = swap_endian(x)

    READ_INT32(magic);
    READ_INT32(num_images);

#undef READ_INT32

    CHECK_EQ(magic, 0x801)
        << "the magic number of labels should be 0x801, i.e., 2049";

    std::vector<uint8_t> res(num_images, 0);
    CHECK_EQ(fread(&res[0], sizeof(uint8_t), res.size(), f), res.size());

    fclose(f);

    return res;
}

std::vector<cnn::Array<uint8_t>> load_images(const std::string& _filename)
{
    auto filename = g_path + "/" + _filename;
    FILE* f = fopen(filename.c_str(), "r");
    if (!f)
    {
        LOG(FATAL) << "failed to open " << filename;
    }
#define READ_INT32(x)                               \
    int x;                                          \
    CHECK_EQ(fread(&x, sizeof(int32_t), 1, f), 1);  \
    x = swap_endian(x)

    READ_INT32(magic);
    READ_INT32(num_images);
    READ_INT32(height);
    READ_INT32(width);

#undef READ_INT32

    CHECK_EQ(magic, 0x803)
        << "the magic number of images should be 0x803, i.e., 2051";

    std::vector<cnn::Array<uint8_t>> res;

    for (int i = 0; i < num_images; i++)
    {
        cnn::Array<uint8_t> img;
        img.init(1, 1, height, width);

        CHECK_EQ(fread(&img[0], sizeof(uint8_t), img.total_, f), img.total_);
        res.emplace_back(std::move(img));
    }

    fclose(f);

    return res;
}

template<typename Dtype>
void data_callback(const std::vector<cnn::Array<Dtype>*> &top)
{
    static int k = 0;
    int n = top[0]->n_;
    CHECK_LE(n, g_train_images.size())
        << "the batch size cannot be larger than the dataset size";

    CHECK_EQ(top[0]->c_, g_train_images[0].c_);
    CHECK_EQ(top[0]->h_, g_train_images[0].h_);
    CHECK_EQ(top[0]->w_, g_train_images[0].w_);

    int stride = top[0]->total_/n;
    CHECK_EQ(stride, g_train_images[0].total_);

    static int epoch = 1;

    for (int i = 0; i < n; i++)
    {
        if (k >= g_train_images.size())
        {
            k = 0;
            epoch++;
            LOG(WARNING) << "epoch is " << epoch;
        }

        int target_index = g_data_index[k];

        const auto& img = g_train_images[target_index];

        for (int j = 0; j < stride; j++)
        {
            top[0]->d_[i*stride + j] = img[j] / Dtype(255);
        }

        if (top.size() == 2)
        {
            top[1]->d_[i] = g_train_labels[target_index];
        }

        k++;
    }
}

// refer to http://www.cplusplus.com/reference/algorithm/random_shuffle/
// random generator function:
int myrandom (int i) { return std::rand()%i;}

void do_training()
{
    g_train_images = load_images(g_train_image_name);
    g_train_labels = load_labels(g_train_label_name);
    CHECK_EQ(g_train_images.size(), g_train_labels.size());

    for (int i = 0; i < g_train_images.size(); i++)
    {
        g_data_index.push_back(i);
    }
    std::srand(time(0));
    std::random_shuffle(g_data_index.begin(), g_data_index.end(), myrandom);

    std::string filename = "../examples/mnist/optimizer.prototxt";

    cnn::Optimizer<double> opt(filename);
    opt.register_data_callback(data_callback);
    opt.start_training();
}

template<typename Dtype>
int argmax(const std::vector<Dtype>& vec)
{
    int max_index = 0;
    auto max_val = vec[0];
    for (int i = 1; i < vec.size(); i++)
    {
        if (vec[i] > max_val)
        {
            max_val = vec[i];
            max_index = i;
        }
    }
    return max_index;
}

void do_testing()
{
#if 1
    g_test_images = load_images(g_test_image_name);
    g_test_labels = load_labels(g_test_label_name);
#else
    g_test_images = load_images(g_train_image_name);
    g_test_labels = load_labels(g_train_label_name);
#endif

    std::ostringstream ss;
    ss << "\n";

    std::string filename = "../examples/mnist/model_for_deploy.prototxt";
    std::string trained = "./trained-bin.prototxt";
    cnn::Network<double> network(filename);
    network.copy_trained_network(trained, true);
    network.reshape();

    int correct = 0;
    int total = 0;

    for (int i = 0; i < 1000; i++)
    // for (int i = 0; i < g_test_images.size(); i++)
    {
        auto& data = *network.get_data_top_mutable(0)[0];
        const auto& img = g_test_images[i];

        CHECK_EQ(data.total_, img.total_);

        for (int k = 0; k < data.total_; k++)
        {
            data[k] = img[k] / (double)255;
        }

        network.perform_predication();
        auto predictions = network.get_predications();
        auto predicted = argmax(predictions);
        if (g_test_labels[i] != predicted)
        {
            LOG(INFO) << (int)g_test_labels[i] << " -> " << predicted;
        }
        else
        {
            correct++;
        }
        total++;

        if (i%500 == 0)
        {
            LOG(INFO) << i << ": current accuracy: " << 100.*correct/total << "\%\n";
        }
    }
    LOG(INFO) << "accuracy: " << 100.*correct / total  << "\%\n";
}

}

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    if (argc == 2)
    {
        do_testing();
    }
    else
    {
        do_training();
    }

    return 0;
}
