
|<sub>Mac OS/Linux</sub>|
|:---:|
|[![Build Status](https://travis-ci.com/csukuangfj/OpenCNN.svg?branch=master)](https://travis-ci.com/csukuangfj/OpenCNN)|

# OpenCNN

OpenCNN is a convolutional neural network framework implemented
with C++11 from scratch.

## Table of contents

- [Features](#features)
- [Supported Layers](#supported-layers)
- [Build](#build)
- [Example with MNIST](#example-with-mnist)
- [Usage](#usage)
- [TODO](#todo)
- [License](#License)

## Features
- Easy to understand
    * Simply implemented and a good source for learning CNN
- Easy to extend
    * Well defined interface for adding new layer types
- Few dependencies
    * Depends only on [protobuf][1], [glog][2] and [gflags][3]
- Fully tested
    * Every layer is covered by unit test with [googletest][4]
    * [autodiff][5] (in forward mode) is implemented to verify the correctness of forward/backward propagation
- Pure C++
    * If you are a big fan of C++
- Runs on CPU
    * No GPU is needed.
    * 95.21% accuracy on MNIST test dataset in 5000 iterations with a batch size of 16

## Supported Layers
- convolutional
- batch normalization
- ReLU
- max pooling
- full connected
- dropout
- softmax
- cross entropy loss (i.e., negative log loss)
- softmax with cross entropy loss
- L2 loss

## Build
### Install Dependencies on Linux (Ubuntu)

```sh
sudo apt-get install libprotobuf-dev protobuf-compiler libgflags-dev libgoogle-glog-dev
```

### Install Dependencies on Mac OS X

```
brew install gflags glog protobuf
```

### Compile From Source

```sh
git clone https://github.com/csukuangfj/OpenCNN.git
cd OpenCNN
mkdir build
cd build
cmake ..
make
```

### Run Unit Test

```sh
cd OpenCNN/build
./gtest
```

It should pass all the test cases on your system.

## Example with MNIST
We use the following network architecture for MNIST:

| Layers                | Description                      |
|-----------------------|----------------------------------|
| Input                 | dim: 1x28x28                     |
| Convolution-1         | num_output: 32, kernel_size: 3x3 |
| Batch normalization-1 |                                  |
| ReLU-1                |                                  |
| Convolution-2         | num_output: 32, kernel_size: 3x3 |
| Batch normalization-2 |                                  |
| ReLU-2                |                                  |
| Max pooling-1         | win_size: 2x2, stride: 2x2       |
| Convolution-3         | num_output:64, kernel_size: 3x3  |
| Batch normalization-3 |                                  |
| ReLU-3                |                                  |
| Convolution-4         | num_output: 64, kernel_size: 3x3 |
| Batch normalization-4 |                                  |
| ReLU-4                |                                  |
| Max pooling-2         | win_size: 2x2, stride: 2x2       |
| Full connected-1      | num_output: 512                  |
| Batch normalization-5 |                                  |
| ReLU-5                |                                  |
| Dropout-1             | keep_prob: 0.8                   |
| Full connected-2      | num_output: 10                   |
| Softmax with log loss |                                  |

During the training a batch size of 16 is used and the accuracy
reaches 95.21% after 5000 iterations. The results for training loss and
test accuracy are plotted in the following figure:

![training-loss-test-accuracy-versus-iterations][6]

A pretrained model taken after 20000 iterations achieving an accuracy
of 96.74% is provided in [OpenCNN-Models][8].

## Usage
Please refer to [examples/mnist][7] for how to use OpenCNN.

More tutorials will be provided later.

## TODO
- [ ] Add advanced optimizers
- [ ] Add more layer types
- [ ] Make code run faster
- [ ] Tutorials and documentation


## License


[8]: https://github.com/csukuangfj/OpenCNN-Models/tree/master/mnist
[7]: /examples/mnist
[6]: /examples/mnist/loss-accuracy-iter.png
[5]: https://en.wikipedia.org/wiki/Automatic_differentiation
[4]: https://github.com/abseil/googletest
[3]: https://github.com/gflags/gflags
[2]: https://github.com/google/glog
[1]: https://github.com/protocolbuffers/protobuf

