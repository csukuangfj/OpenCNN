#!/bin/bash

cd build
. ../examples/mnist/data/download_and_unzip_data.sh
wget https://raw.githubusercontent.com/csukuangfj/OpenCNN-Models/master/mnist/mnist-bin.prototxt-20000
./mnist 0

# you should see the test accuracy at the end of the output

