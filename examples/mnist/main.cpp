#include <glog/logging.h>

#include <string>
#include <fstream>
#include <vector>

#include "cnn/array.hpp"
#include "cnn/io.hpp"

namespace
{

std::string g_path = "../examples/mnist/data";
std::string g_train_image_name = "train-images-idx3-ubyte";
std::string g_train_label_name = "train-labels-idx1-ubyte";

std::vector<std::pair<int, cnn::Array<uint8_t>>> g_train_data;

int swap_endian(int in)
{
    int res = in;
    char* c = (char*)&res;

    std::swap(c[0], c[3]);
    std::swap(c[1], c[2]);
    return res;
}

void load_labels(const std::string& _filename)
{

}

void load_images(const std::string& _filename)
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

    LOG(INFO) << "there are " << num_images << " images";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: " << width;

    for (int i = 0; i < 5; i++)
    {
        std::ostringstream ss;
        ss << i << ".pgm";
        cnn::Array<uint8_t> img;
        img.init(1, 1, height, width);
        CHECK_EQ(fread(&img[0], sizeof(uint8_t), img.total_, f), img.total_);
        cnn::write_pgm(ss.str(), img);
    }

    fclose(f);
}

}

int main(int /*argc*/, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    load_images(g_train_image_name);

    return 0;
}
