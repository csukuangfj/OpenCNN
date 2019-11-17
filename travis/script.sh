#!/bin/bash

# Copyright 2019. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

CNN_ROOT_DIR=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

cd $CNN_ROOT_DIR

./travis/install-bazel.sh

# bazel test --config=cpplint //cnn/... -s

bazel test //cnn/... -s
