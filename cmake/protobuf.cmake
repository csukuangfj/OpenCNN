
#[==[
refer to
https://cmake.org/cmake/help/v3.9/module/FindProtobuf.html
]==]
include(FindProtobuf)
find_package(Protobuf REQUIRED)

message(STATUS "protobuf include dirs: ${Protobuf_INCLUDE_DIR}")
message(STATUS "protobuf libs: ${Protobuf_LIBRARY}")

include_directories(${Protobuf_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

