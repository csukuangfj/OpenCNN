#!/bin/bash

#
# Download mnist data and unzip them
#

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

base_url="http://yann.lecun.com/exdb/mnist"

filenames=(
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
)

function download_and_decompress()
{
    filename="${CUR_DIR}/$1"

    if [ -e "${filename}" ]; then
        return
    fi

    if [ ! -e "${filename}.gz" ]; then
        url="${base_url}/$1.gz"
        echo "Downloading ${url} to ${filename}.gz"
        wget "${url}" -O "${filename}.gz"
    fi
    gunzip "${filename}.gz"

}

for filename in ${filenames[*]}
do
    download_and_decompress ${filename}
done


