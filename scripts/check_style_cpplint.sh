#!/bin/bash

# Copyright 2018-2020. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

bold='\033[1m'
red='\033[31m'
green='\033[32m'
default='\033[0m'

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
cnn_dir=$cur_dir/..
build_dir=$cnn_dir/build
cpplint_src=$build_dir/third_party/cpplint/src/cpplint.py

function ok() {
  printf "${bold}${green}[OK]${default} $1\n"
}

function error() {
  printf "${bold}${red}[FAILED]${default} $1\n"
  exit 1
}

# return true if the given file is a c++ source file
# return false otherwise
function is_source_code_file() {
  case "$1" in
    *.cc|*.h)
      echo true;;
    *)
      echo false;;
  esac
}



function check_style() {
  python3 $cpplint_src $1 || error $1
}


cd $cnn_dir

files=$(git status -s -uno --porcelain | awk '{print $2}')

for f in $files; do
  need_check=$(is_source_code_file $f)
  if $need_check; then
    check_style $f
  fi
done

ok "Check passed."
