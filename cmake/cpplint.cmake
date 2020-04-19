# Copyright 2018-2020. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

# Download cpplint.py from GitHub

include(ExternalProject)

function(download_cpplint)
  set(cpplint_URL "https://raw.githubusercontent.com/cpplint/cpplint/master/cpplint.py")
  set(cpplint_DIR "${cnn_THIRD_PARTY_DIR}/cpplint")

  ExternalProject_Add(
    cpplint
    URL                 ${cpplint_URL}
    DOWNLOAD_NO_EXTRACT NO
    TIMEOUT             10
    PREFIX              ${cpplint_DIR}
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
    TEST_COMMAND        ""
    LOG_DOWNLOAD        ON
    LOG_CONFIGURE       ON
  )

  ExternalProject_Get_Property(cpplint SOURCE_DIR)
  set(cpplint_SOURCE_DIR ${SOURCE_DIR} PARENT_SCOPE)
endfunction()

download_cpplint()
