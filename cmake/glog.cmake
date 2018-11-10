#
# GLOG_INCLUDE_DIRS
# GLOG_LIBRARIES
#

find_path(GLOG_INCLUDE_DIRS
        glog/logging.h
        PATHS
        /usr/include
        DOC "Path to glog/logging.h"
)
if (NOT GLOG_INCLUDE_DIRS)
    message(FATAL_ERROR "Could not find glog/logging.h")
endif()

find_library(GLOG_LIBRARIES
        NAMES libglog.so glog
        PATHS
        /usr/lib/x86_64-linux-gnu
        DOC "Path to libglog.so"
        )
if (NOT GLOG_LIBRARIES)
    message(FATAL_ERROR "Could not find libglog.so")
endif()

include_directories(${GLOG_INCLUDE_DIRS})
message(STATUS "GLOG_INCLUDE_DIRS: ${GLOG_INCLUDE_DIRS}")
message(STATUS "GLOG_LIBRARIES: ${GLOG_LIBRARIES}")

#[[
brew install glog
]]
