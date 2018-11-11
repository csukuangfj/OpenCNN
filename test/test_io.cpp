#include <gtest/gtest.h>

#include "cnn/common.hpp"
#include "cnn/io.hpp"

namespace cnn
{

TEST(io_test, txt)
{
    ArrayProto arr;
    arr.set_n(1);
    arr.set_c(2);
    arr.set_h(3);
    arr.set_w(4);
    for (int i = 0; i < 24; i++)
    {
        arr.add_d(i);
    }
    write_proto_txt("a.txt", arr);

    ArrayProto arr2;
    read_proto_txt("a.txt", &arr2);

    EXPECT_EQ(arr.n(), arr2.n());
    EXPECT_EQ(arr.c(), arr2.c());
    EXPECT_EQ(arr.h(), arr2.h());
    EXPECT_EQ(arr.w(), arr2.w());
    EXPECT_EQ(arr.d_size(), arr2.d_size());
    for (int i = 0; i < arr.d_size(); i++)
    {
        EXPECT_EQ(arr.d(i), arr2.d(i));
    }
}

TEST(io_test, bin)
{
    ArrayProto arr;
    arr.set_n(1);
    arr.set_c(2);
    arr.set_h(3);
    arr.set_w(4);
    for (int i = 0; i < 24; i++)
    {
        arr.add_d(i);
    }
    write_proto_bin("a.bin", arr);

    ArrayProto arr2;
    read_proto_bin("a.bin", &arr2);

    EXPECT_EQ(arr.n(), arr2.n());
    EXPECT_EQ(arr.c(), arr2.c());
    EXPECT_EQ(arr.h(), arr2.h());
    EXPECT_EQ(arr.w(), arr2.w());
    EXPECT_EQ(arr.d_size(), arr2.d_size());
    for (int i = 0; i < arr.d_size(); i++)
    {
        EXPECT_EQ(arr.d(i), arr2.d(i));
    }
}

}  // namespace cnn
