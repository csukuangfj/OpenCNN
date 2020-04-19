# Copyright 2018-2020. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

# in-source build is not recommended.
message(STATUS ${CMAKE_SOURCE_DIR})
message(STATUS ${CMAKE_BINARY_DIR})
if("x${CMAKE_SOURCE_DIR}" STREQUAL "x${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "\
In-source build is not a good practice.
Please use:
  mkdir build
  cd build
  cmake ..
to build this project"
  )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
)
