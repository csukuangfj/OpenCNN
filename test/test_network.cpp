#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sstream>

#define private public

#include "cnn/array_math.hpp"
#include "cnn/network.hpp"
#include "proto/cnn.pb.h"

namespace cnn
{

template<typename Dtype>
class NetworkTest : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        filename_ = "../proto/model.prototxt";
    }

    Dtype func(int x)
    {
        static Dtype w0 = 25.5;
        static Dtype w1 = 10;
        static Dtype w2 = 0.05;
        static Dtype w3 = 0.002;
        return w0 + w1*x + w2*x*x + w3*x*x*x;
    }

    std::vector<std::pair<std::vector<Dtype>, Dtype>>
    generate_test_data()
    {
        // y = w0 + w1*x + w2*x^2 + w3*x^3
        int low = -20;
        int high = 20;
        std::set<int> x;

        std::vector<std::pair<std::vector<Dtype>, Dtype>> res;

        while(x.size() < 10)
        {
            int i = uniform(low, high);
            if (x.count(i)) continue;

            x.insert(i);
            auto p = std::make_pair<std::vector<Dtype>, Dtype>(
                            {Dtype(i), Dtype(i)*i, Dtype(i)*i*i}, func(i));
            res.emplace_back(p);
        }

        return res;
    }
protected:
    std::string filename_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(NetworkTest, MyTypes);

TYPED_TEST(NetworkTest, constructor)
{
    Network<TypeParam> network(this->filename_);
    network.reshape();
    for (const auto& p : network.data_)
    {
        LOG(INFO) << p.first << ": " << p.second->shape_info();
    }
}

TYPED_TEST(NetworkTest, fprop1)
{
    Network<TypeParam> network(this->filename_);
    network.reshape();

    auto input = network.get_data_output(0);
    input[0]->operator()(0, 0, 0, 0) = 1;
    input[0]->operator()(0, 0, 0, 1) = 2;
    input[0]->operator()(0, 0, 0, 2) = 3;

    input[0]->operator()(1, 0, 0, 0) = 4;
    input[0]->operator()(1, 0, 0, 1) = 5;
    input[0]->operator()(1, 0, 0, 2) = 6;

    input[1]->d_[0] = 100;
    input[1]->d_[1] = 200;

    auto param = network.layer(1)->param();
    // test only one output
    param[0]->d_[0] = 10;
    param[0]->d_[1] = 20;
    param[0]->d_[2] = 30;

    param[1]->d_[0] = 2;

    network.fprop();

    auto output = network.get_data_output(1)[0];
    // 1*10+2*20+3*30+2 = 10+40+90+2=142
    EXPECT_EQ(output->d_[0], 142);

    // 40+100+180+2=322
    EXPECT_EQ(output->d_[1], 322);

    auto loss = network.get_data_output(2)[0]->d_[0];
    TypeParam expected_loss;
    expected_loss = (142-100)*(142-100) + (322-200)*(322-200);
    // EXPECT_EQ(loss, expected_loss/2);
}

TYPED_TEST(NetworkTest, fprop2)
{
    auto data = this->generate_test_data();
    std::ostringstream ss;
#if 0
    for (const auto& p : data)
    {
        for (const auto& x: p.first)
        {
            ss << x << " ";
        }
        ss << p.second << "\n";
    }
    LOG(INFO) << "\n" << ss.str();
#endif

    Network<TypeParam> network(this->filename_);
    network.reshape();

    for (const auto& p : network.data_)
    {
        LOG(INFO) << p.first << ": " << p.second->shape_info();
    }

    auto input = network.get_data_output(0);

    for (int n = 0; n < input[0]->n_; n++)
    {
        for (int i = 0; i < input[0]->w_; i++)
        {
            input[0]->operator()(n, 0, 0, i) = (data[n].first)[i];
        }
        input[1]->d_[n] = data[n].second;
    }

    ss.str("");
    int k = 0;
    for (int i = 0; i < input[0]->total_; i++)
    {
        if ((i%input[0]->w_ == 0) && i)
        {
            ss << input[1]->d_[k];
            k++;
            ss << "\n";
        }
        ss << input[0]->d_[i] << " ";
    }

    ss << input[1]->d_[k];
    ss << "\n";
    LOG(INFO) << ss.str();

    network.fprop();

    auto loss = network.get_data_output(2)[0]->d_[0];
    LOG(WARNING) << "loss is " << loss;

}

} // namespace cnn
