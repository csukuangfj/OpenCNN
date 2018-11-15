#pragma once

#include <glog/logging.h>

#include "proto/cnn.pb.h"

// TODO(fangjun): avoid define here!
// we use the same threshold as caffe for computing log()
#define g_log_threshold (1e-20)
