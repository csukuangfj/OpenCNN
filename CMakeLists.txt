
cmake_minimum_required(VERSION 3.0)
project(cnn)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/compiler-flags.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/protobuf.cmake)

include_directories(include)

add_subdirectory(proto)

add_library(
    core STATIC
    src/io.cpp
    src/rng.cpp
    # src/array.cpp
    # src/layer.cpp
    # src/full_connected_layer.cpp
    # src/input_layer.cpp
    # src/network.cpp
    # src/l2_loss_layer.cpp
    # src/array_math.cpp
    # src/optimizer.cpp
    # src/softmax_layer.cpp
    # src/log_loss_layer.cpp
    # src/softmax_with_log_loss_layer.cpp
    )
target_link_libraries(
    core
    cnn_proto
    ${GLOG_LIBRARIES}
    )

add_subdirectory(test)

add_subdirectory(tools)
add_subdirectory(examples)

