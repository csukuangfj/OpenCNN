
protobuf_generate_cpp(
    PROTO_SRCS
    PROTO_HDRS
    cnn.proto
    )

add_library(
    cnn_proto SHARED
    ${PROTO_SRCS}
    ${PROTO_HDRS}
    )

target_link_libraries(
    cnn_proto
    ${MY_PROTOBUF_LIBS}
    )

