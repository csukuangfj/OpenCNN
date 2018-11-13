
#pragma once

#include <google/protobuf/message.h>

#include <string>

namespace cnn
{

void read_proto_txt(
        const std::string& filename,
        google::protobuf::Message* message);

void write_proto_txt(
        const std::string& filename,
        const google::protobuf::Message& message);

void read_proto_bin(
        const std::string& filename,
        google::protobuf::Message* message);

void write_proto_bin(
        const std::string& filename,
        const google::protobuf::Message& message);

void string_to_proto(
        const std::string& filename,
        google::protobuf::Message* message);

}  // namespace cnn

