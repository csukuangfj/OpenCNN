
#pragma once

#include <google/protobuf/message.h>

#include <string>
#include "cnn/array.hpp"

namespace cnn {

void read_proto_txt(const std::string& filename,
                    google::protobuf::Message* message);

void write_proto_txt(const std::string& filename,
                     const google::protobuf::Message& message);

void read_proto_bin(const std::string& filename,
                    google::protobuf::Message* message);

void write_proto_bin(const std::string& filename,
                     const google::protobuf::Message& message);

void string_to_proto(const std::string& filename,
                     google::protobuf::Message* message);

void write_pgm(const std::string& filename, const Array<uint8_t>& img);

void read_pgm(const std::string& filename, Array<uint8_t>* img);

}  // namespace cnn
