# Copyright 2018-2020. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(
  cnn
  VERSION     0.1
  DESCRIPTION "An Open Convolutional Neural Network Framework in C++ From Scratch"
  LANGUAGES   CXX C
)

set(cnn_THIRD_PARTY_DIR
  "${CMAKE_BINARY_DIR}/third_party"
  CACHE STRING
  "Directory to save & build downloaded third party libraries"
)

set(CMAKE_BUILD_TYPE Release)

include(cmake/paths.cmake) # it will setup CMAKE_MODUEL_PATH
include(cpplint)
include(googletest)

enable_testing()

add_subdirectory(cnn)
