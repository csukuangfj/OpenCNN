// Copyright 2018-2020. All Rights Reserved.
// Author: csukuangfj@gmail.com (Fangjun Kuang)

#include "gtest/gtest.h"

#include "cnn/autodiff/memory_pool.h"

namespace cnn {

TEST(MemoryArena, TestAllocate) {
  MemoryArena<4> arena(10);
  int* p1 = reinterpret_cast<int*>(arena.Allocate(1));
  *p1 = 10;
  int* p2 = reinterpret_cast<int*>(arena.Allocate(1));
  *p2 = 20;

  EXPECT_EQ(p1[0], 10);
  EXPECT_EQ(p1[1], 20);
}

}  // namespace cnn
