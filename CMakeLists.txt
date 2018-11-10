
cmake_minimum_required(VERSION 3.0)
project(cnn)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

add_subdirectory(test)

