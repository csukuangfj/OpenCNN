
#[==[
refer to
https://cmake.org/cmake/help/v3.9/module/FindProtobuf.html
]==]
find_package(Protobuf REQUIRED)

set(MY_PROTOBUF_INCLUDE_DIR)
set(MY_PROTOBUF_LIBS)
if(APPLE)
    set(MY_PROTOBUF_INCLUDE_DIR ${Protobuf_INCLUDE_DIRS})
    set(MY_PROTOBUF_LIBS ${Protobuf_LIBRARIES})
elseif(UNIX)
    set(MY_PROTOBUF_INCLUDE_DIR ${PROTOBUF_INCLUDE_DIR})
    set(MY_PROTOBUF_LIBS ${PROTOBUF_LIBRARY})
endif()

message(STATUS "protobuf include dirs: ${MY_PROTOBUF_INCLUDE_DIR}")
message(STATUS "protobuf libs: ${MY_PROTOBUF_LIBS}")

include_directories(${MY_PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

