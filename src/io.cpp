
#include <fcntl.h>

#include <glog/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>  // NOLINT
#include <string>

#include "cnn/io.hpp"

namespace cnn {
// modified from
// https://stackoverflow.com/questions/10842066/parse-in-text-file-for-google-protocol-buffer
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/io.cpp

void read_proto_txt(const std::string& filename,
                    google::protobuf::Message* message) {
  int fd = open(filename.c_str(), O_RDONLY);
  CHECK_GT(fd, 0) << "Failed to open " << filename;

  google::protobuf::io::FileInputStream fstream(fd);
  CHECK(google::protobuf::TextFormat::Parse(&fstream, message))
      << "Failed to read " << filename;

  close(fd);
}

void write_proto_txt(const std::string& filename,
                     const google::protobuf::Message& message) {
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CHECK_GT(fd, 0) << "Failed to create " << filename;

  google::protobuf::io::FileOutputStream fstream(fd);
  fstream.SetCloseOnDelete(true);
  CHECK(google::protobuf::TextFormat::Print(message, &fstream))
      << "Failed to write " << filename;
}

// refer to
// https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.coded_stream
void read_proto_bin(const std::string& filename,
                    google::protobuf::Message* message) {
  int fd = open(filename.c_str(), O_RDONLY);
  CHECK_GT(fd, 0) << "Failed to open " << filename;

  google::protobuf::io::ZeroCopyInputStream* raw_input =
      new google::protobuf::io::FileInputStream(fd);
  google::protobuf::io::CodedInputStream coded_input(raw_input);
  CHECK(message->ParseFromCodedStream(&coded_input))
      << "Failed to read " << filename;

  delete raw_input;
  close(fd);
}

// refer to https://github.com/BVLC/caffe/blob/master/src/caffe/util/io.cpp#L67
void write_proto_bin(const std::string& filename,
                     const google::protobuf::Message& message) {
  std::ofstream fstream(filename,
                        std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK(message.SerializeToOstream(&fstream)) << "Failed to write " << filename;
}

void string_to_proto(const std::string& model,
                     google::protobuf::Message* message) {
  CHECK(google::protobuf::TextFormat::ParseFromString(model, message));
}

// refer to http://netpbm.sourceforge.net/doc/pgm.html
// for the spec of pgm
void write_pgm(const std::string& filename, const Array<uint8_t>& img) {
  std::ofstream of(filename,
                   std::ios::out | std::ios::trunc | std::ios::binary);
  if (!of) {
    LOG(FATAL) << "cannot create file " << filename;
  }

  of << "P5"
     << "\n";

  of << "# created with cnn"
     << "\n";

  of << img.w_ << " " << img.h_ << "\n";

  of << 255 << "\n";

  of.write((const char*)&img[0], img.total_);
}

void read_pgm(const std::string& filename, Array<uint8_t>* img) {
  // std::ifstream fs(filename, std::ios::binary);
  std::ifstream fs(filename);
  if (!fs) {
    LOG(FATAL) << "cannot read file " << filename;
  }

  std::string format;
  std::getline(fs, format);
  LOG(INFO) << "format is: " << format;

  std::string comments;
  std::getline(fs, comments);

  int width, height;
  fs >> width >> height;

  int max_val;
  fs >> max_val;

  CHECK_EQ(max_val, 255);

  fs.get();  // eat the new line \n

  img->init(1, 1, height, width);

  fs.read(reinterpret_cast<char*>(&img[0][0]), img->total_);
}

}  // namespace cnn
