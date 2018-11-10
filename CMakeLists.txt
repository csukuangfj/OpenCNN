
cmake_minimum_required(VERSION 3.0)
project(cnn)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/compiler-flags.cmake)

include_directories(include)

add_library(core
    src/array.cpp
    )
target_link_libraries(
    core
    ${GLOG_LIBRARIES}
    )

add_subdirectory(test)

