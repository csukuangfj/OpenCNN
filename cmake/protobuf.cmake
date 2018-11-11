
#[==[
refer to
https://cmake.org/cmake/help/v3.9/module/FindProtobuf.html
]==]
find_package(Protobuf REQUIRED)
message(STATUS "protobuf include dirs: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "protobuf libs: ${Protobuf_LIBRARIES}")

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

