#include <gtest/gtest.h>

#include "cnn/array.hpp"
#include "cnn/common.hpp"
#include "cnn/io.hpp"

#include "proto/cnn.pb.h"

namespace cnn {

TEST(io_test, txt) {
  ArrayProto arr;
  arr.set_n(1);
  arr.set_c(2);
  arr.set_h(3);
  arr.set_w(4);
  for (int i = 0; i < 24; i++) {
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
  for (int i = 0; i < arr.d_size(); i++) {
    EXPECT_EQ(arr.d(i), arr2.d(i));
  }
}

TEST(io_test, bin) {
  ArrayProto arr;
  arr.set_n(1);
  arr.set_c(2);
  arr.set_h(3);
  arr.set_w(4);
  for (int i = 0; i < 24; i++) {
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
  for (int i = 0; i < arr.d_size(); i++) {
    EXPECT_EQ(arr.d(i), arr2.d(i));
  }
}

TEST(io_test, string_to_proto) {
  const char* model =
      R"proto(
    layer_proto {
      name: "input"
      type: INPUT
      top: "data"
      top: "label"
      input_proto { n: 5 c: 1 h: 1 w: 1 }
    }
      )proto";
  NetworkProto proto;
  string_to_proto(model, &proto);
  LOG(INFO) << proto.DebugString();
}

TEST(io_test, write_read_pgm) {
  std::string filename = "abc.pgm";
  Array<uint8_t> a;
  a.init(1, 1, 50, 50);
  LOG(INFO) << "a.h: " << a.h_;
  for (int i = 0; i < a.h_; i++) {
    a(0, 0, i, i) = 255;
  }
  write_pgm(filename, a);

  Array<uint8_t> b;
  read_pgm(filename, &b);

  EXPECT_TRUE(a.has_same_shape(b));
  for (int i = 0; i < a.total_; i++) {
    EXPECT_EQ(a[i], b[i]);
  }
}

}  // namespace cnn
