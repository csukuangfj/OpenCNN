#!/bin/bash

# Copyright 2019. All Rights Reserved.
# Author: fangjun.kuang@gmail.com (Fangjun Kuang)

BAZEL_VERSION=0.25.0

if which bazel >/dev/null; then
  # get bazel version
  version=$(bazel version 2>/dev/null | grep "Build label" | awk '{print $NF}')
  if [[ $version == $BAZEL_VERSION ]]; then
    echo "bazel $BAZEL_VERSION has already been installed, skip"
    exit 0
  else
    echo "Replace bazel $version with $BAZEL_VERSION"
  fi
fi

os=$(uname -s | tr 'A-Z' 'a-z')
if [[ $os != darwin && $os != linux ]]; then
  echo "We support only Darwin and Linux"
  echo "Current OS is: $(uname -s)"
  exit 1
fi

filename=bazel-$BAZEL_VERSION-installer-$os-x86_64.sh

curl -L -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/$filename
chmod +x $filename
./$filename --user

rm $filename
echo "Done. Installed to $(which bazel)"
