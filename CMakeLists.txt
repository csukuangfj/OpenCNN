
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
    src/array.cpp
    src/io.cpp
    )
target_link_libraries(
    core
    cnn_proto
    ${GLOG_LIBRARIES}
    )

add_subdirectory(test)

