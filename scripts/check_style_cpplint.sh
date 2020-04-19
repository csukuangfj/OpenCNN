#!/bin/bash

# Copyright 2018-2020. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

default='\033[0m'
bold='\033[1m'
red='\033[31m'
green='\033[32m'

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
cnn_dir=$(cd $cur_dir/.. && pwd)
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

function check_last_commit() {
  files=$(git show HEAD --name-only --diff-filter=ACMRTX)
  echo $files
}

function check_current_dir() {
  files=$(git status -s -uno --porcelain | awk '{print $2}')
  echo $files
}

function do_check() {
  if [ $# -eq 1 ]; then
    echo "checking last commit"
    files=$(check_last_commit)
  else
    echo "checking current dir"
    files=$(check_current_dir)
  fi

  for f in $files; do
    need_check=$(is_source_code_file $f)
    if $need_check; then
      check_style $f
    fi
  done
}

function main() {
  if [ ! -f $cpplint_src ]; then
    error "\n$cpplint_src does not exist.\n\
Please run
    mkdir build
    cd build
    cmake ..
before running this script."
  fi

  do_check $1

  ok "Check passed."
}

cd $cnn_dir

main $1
