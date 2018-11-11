
#include <fcntl.h>

#include <glog/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>      // NOLINT
#include <string>

#include "cnn/io.hpp"

namespace cnn
{
// modified from
// https://stackoverflow.com/questions/10842066/parse-in-text-file-for-google-protocol-buffer
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/io.cpp

void read_proto_txt(
        const std::string& filename,
        google::protobuf::Message* message)
{
    int fd = open(filename.c_str(), O_RDONLY);
    CHECK_GT(fd, 0) << "Failed to open " << filename;

    google::protobuf::io::FileInputStream fstream(fd);
    CHECK(google::protobuf::TextFormat::Parse(&fstream, message))
        << "Failed to read " << filename;

    close(fd);
}

void write_proto_txt(
        const std::string& filename,
        const google::protobuf::Message& message)
{
    int fd = open(filename.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK_GT(fd, 0) << "Failed to create " << filename;

    google::protobuf::io::FileOutputStream fstream(fd);
    fstream.SetCloseOnDelete(true);
    CHECK(google::protobuf::TextFormat::Print(message, &fstream))
        << "Failed to write " << filename;
}

// refer to
// https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.coded_stream
void read_proto_bin(
        const std::string& filename,
        google::protobuf::Message* message)
{
    int fd = open(filename.c_str(), O_RDONLY);
    CHECK_GT(fd, 0) << "Failed to open " << filename;

    google::protobuf::io::ZeroCopyInputStream *raw_input =
        new google::protobuf::io::FileInputStream(fd);
    google::protobuf::io::CodedInputStream coded_input(raw_input);
    CHECK(message->ParseFromCodedStream(&coded_input))
        << "Failed to read " << filename;

    delete raw_input;
    close(fd);
}

// refer to https://github.com/BVLC/caffe/blob/master/src/caffe/util/io.cpp#L67
void write_proto_bin(
        const std::string& filename,
        const google::protobuf::Message& message)
{
    std::ofstream fstream(filename,
            std::ios::out | std::ios::trunc | std::ios::binary);
    CHECK(message.SerializeToOstream(&fstream))
        << "Failed to write " << filename;
}

}  // namespace cnn
