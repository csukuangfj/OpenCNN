# Copyright 2019. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def cnn_repositories():
    bazel_federation_commit_id = "130c84ec6d60f31b711400e8445a8d0d4a2b5de8"
    bazel_federation_sha256 = "9d4fdf7cc533af0b50f7dd8e58bea85df3b4454b7ae00056d7090eb98e3515cc"
    http_archive(
        name = "bazel_federation",
        sha256 = bazel_federation_sha256,
        strip_prefix = "bazel-federation-" + bazel_federation_commit_id,
        url = "https://github.com/bazelbuild/bazel-federation/archive/{}.zip".format(bazel_federation_commit_id),
    )

    bazel_skylib_commit_id = "f130d7c388e6beeb77309ba4e421c8f783b91739"
    bazel_skylib_sha256 = "8eb5bce85cddd2f4e5232c94e57799de62b1671ce4f79f0f83e10e2d3b2e7986"
    http_archive(
        name = "bazel_skylib",
        sha256 = bazel_skylib_sha256,
        strip_prefix = "bazel-skylib-" + bazel_skylib_commit_id,
        url = "https://github.com/bazelbuild/bazel-skylib/archive/{}.zip".format(bazel_skylib_commit_id),
    )

    http_archive(
        name = "com_google_googletest",
        sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
        strip_prefix = "googletest-release-1.10.0",
        urls = ["https://github.com/google/googletest/archive/release-1.10.0.tar.gz"],
    )

    google_style_guide_commit_id = "83a9e8d7ca3d47239cb0a7bf532de791e6df3ba6"
    google_style_guide_sha256 = "7ff8e886c1f48754e083137300a843896305fb0677e606f5abe234d3e68f70ee"
    http_archive(
        name = "google_style_guide",
        build_file = "//third_party:google_style_guide.BUILD",
        sha256 = google_style_guide_sha256,
        strip_prefix = "styleguide-" + google_style_guide_commit_id,
        urls = ["https://github.com/google/styleguide/archive/{}.tar.gz".format(google_style_guide_commit_id)],
    )

    http_archive(
        name = "com_google_glog",
        patch_cmds = ["sed -i -e 's/glog_library()/glog_library(with_gflags=0)/' BUILD"],
        sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
        strip_prefix = "glog-0.4.0",
        urls = ["https://github.com/google/glog/archive/v0.4.0.tar.gz"],
    )

    http_archive(
        name = "com_google_protobuf",
        sha256 = "e82ee5bdde198e0a1935e280748a86a7989474ea771418a2fd90f03e2e65b99b",
        strip_prefix = "protobuf-3.10.1",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protobuf-cpp-3.10.1.tar.gz",
        ],
    )

    rules_proto_commit_id = "97d8af4dc474595af3900dd85cb3a29ad28cc313"
    rules_proto_sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208"
    http_archive(
        name = "rules_proto",
        sha256 = rules_proto_sha256,
        strip_prefix = "rules_proto-" + rules_proto_commit_id,
        urls = [
            "https://github.com/bazelbuild/rules_proto/archive/{}.tar.gz".format(rules_proto_commit_id),
        ],
    )
