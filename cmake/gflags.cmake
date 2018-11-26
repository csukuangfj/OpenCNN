#
# GFLAGS_INCLUDE_DIRS
# GFLAGS_LIBRARIES
#

find_path(GFLAGS_INCLUDE_DIRS
        gflags/gflags.h
        PATHS
        /usr/local/include
        $ENV{HOME}/software/gflags/include
        DOC "Path to gflags/gflags.h"
)
if (NOT GFLAGS_INCLUDE_DIRS)
    message(FATAL_ERROR "Could not find gflags/gflags.h")
endif()

find_library(GFLAGS_LIBRARIES
        NAMES libgflags.so libgflags.dylib gflags
        PATHS
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
        $ENV{HOME}/software/gflags/lib
        DOC "Path to libgflags.so"
        )
if (NOT GFLAGS_LIBRARIES)
    message(FATAL_ERROR "Could not find libgflags.so")
endif()

include_directories(${GFLAGS_INCLUDE_DIRS})
message(STATUS "GFLAGS_INCLUDE_DIRS: ${GFLAGS_INCLUDE_DIRS}")
message(STATUS "GFLAGS_LIBRARIES: ${GFLAGS_LIBRARIES}")

#[[
brew install gflags
]]
