# Copyright 2019. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def cnn_repositories():
    rules_cc_commit_id = "c2b692b4e4ee5260e741eb479d3ecb686c89a48d"
    rules_cc_sha256 = "f1abfedd988c56a76cf3f85dc6f0e1dee9f422785bcd1805605fef189171347e"
    http_archive(
        name = "rules_cc",
        sha256 = rules_cc_sha256,
        strip_prefix = "rules_cc-" + rules_cc_commit_id,
        urls = ["https://github.com/bazelbuild/rules_cc/archive/{}.tar.gz".format(rules_cc_commit_id)],
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
