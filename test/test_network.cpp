#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sstream>

#define private public

#include "cnn/array_math.hpp"
#include "cnn/network.hpp"
#include "cnn/io.hpp"
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
        static Dtype w0 = 5;
        static Dtype w1 = 10;
        return w0 + w1*x;
    }

    std::vector<std::pair<std::vector<Dtype>, Dtype>>
    generate_test_data()
    {
        // y = w0 + w1*x
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
                            {Dtype(i)}, func(i));
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
    NetworkProto proto;
    read_proto_txt(this->filename_, &proto);
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_n(2);
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_c(1);
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_h(1);
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_w(3);
    proto.mutable_layer_proto(1)->mutable_fc_proto()->set_num_output(1);

    Network<TypeParam> network(proto);
    network.reshape();

    auto input = network.get_data_top_mutable(0);
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

    auto output = network.get_data_top(1)[0];
    // 1*10+2*20+3*30+2 = 10+40+90+2=142
    EXPECT_EQ(output->d_[0], 142);

    // 40+100+180+2=322
    EXPECT_EQ(output->d_[1], 322);

    auto loss = network.get_data_top(2)[0]->d_[0];
    TypeParam expected_loss;
    expected_loss = (142-100)*(142-100) + (322-200)*(322-200);
    EXPECT_EQ(loss, expected_loss/2);
}

TYPED_TEST(NetworkTest, fprop2)
{
    auto data = this->generate_test_data();
    std::ostringstream ss;
#if 1
    ss << "\n";
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

    NetworkProto proto;
    read_proto_txt(this->filename_, &proto);
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_n(data.size());
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_c(1);
    proto.mutable_layer_proto(0)->mutable_input_proto()->set_h(1);

    proto.mutable_layer_proto(0)->mutable_input_proto()->set_w(
            data[0].first.size());

    proto.mutable_layer_proto(1)->mutable_fc_proto()->set_num_output(1);

    Network<TypeParam> network(proto);
    network.reshape();

    for (const auto& p : network.data_)
    {
        LOG(INFO) << p.first << ": " << p.second->shape_info();
    }

    auto input = network.get_data_top_mutable(0);

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

    TypeParam learning_rate = 1e-4;

    auto&w = *network.layer(1)->mutable_param()[0];
    auto&b = *network.layer(1)->mutable_param()[1];

    auto&dw = *network.layer(1)->mutable_gradient()[0];
    auto&db = *network.layer(1)->mutable_gradient()[1];
    for (int i = 0; i < 6000; i++)
    {
        network.fprop();
        auto loss = network.get_data_top(2)[0]->d_[0];

        ss.str("");
        ss << "before iteration: " << i << "\n";
        ss << "b: ";
        for (int k = 0; k < w.total_; k++)
        {
            ss << b[k] << " ";
        }
        ss << "\nw: ";
        for (int k = 0; k < w.total_; k++)
        {
            ss << w[k] << " ";
        }
        ss << "\n----loss: " << loss << "---\n";

        network.bprop();

        ax_plus_by<TypeParam>(
                w.total_,
                -learning_rate,
                &dw[0],
                1,
                &w[0]);

        ax_plus_by<TypeParam>(
                b.total_,
                -learning_rate,
                &db[0],
                1,
                &b[0]);

        ss << "after iteration: " << i << "\n";
        ss << "b: ";
        for (int k = 0; k < w.total_; k++)
        {
            ss << b[k] << " ";
        }
        ss << "\nw: ";
        for (int k = 0; k < w.total_; k++)
        {
            ss << w[k] << " ";
        }

        if (i%1000 == 0)
        {
            LOG(INFO) << ss.str();
        }
    }
}

} // namespace cnn
