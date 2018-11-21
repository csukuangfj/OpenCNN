#include <glog/logging.h>

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

std::vector<cnn::Array<uint8_t>> g_train_images;
std::vector<uint8_t> g_train_labels;

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

        const auto& img = g_train_images[k];

        for (int j = 0; j < stride; j++)
        {
            top[0]->d_[i*stride + j] = img[j] / Dtype(255);
        }

        if (top.size() == 2)
        {
            top[1]->d_[i] = g_train_labels[k];
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

    g_train_images = load_images(g_train_image_name);
    g_train_labels = load_labels(g_train_label_name);
    CHECK_EQ(g_train_images.size(), g_train_labels.size());

    std::string filename = "../examples/mnist/optimizer.prototxt";

    cnn::Optimizer<double> opt(filename);
    opt.register_data_callback(data_callback);
    opt.start_training();

    return 0;
}
