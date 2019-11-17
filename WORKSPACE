# Copyright 2019. All Rights Reserved.
# Author: csukuangfj@gmail.com (Fangjun Kuang)

workspace(name = "cnn")

load("//cnn:workspace.bzl", "cnn_repositories")

cnn_repositories()

load(
    "@bazel_federation//:repositories.bzl",
    "rules_cc",
    "rules_java",
    "rules_python",
)

#==================== rules_cc ====================
rules_cc()

load("@bazel_federation//setup:rules_cc.bzl", "rules_cc_setup")

rules_cc_setup()

#==================== rules_java ====================
rules_java()

load("@bazel_federation//setup:rules_java.bzl", "rules_java_setup")

rules_java_setup()

#==================== rules_python ====================
rules_python()

load("@bazel_federation//setup:rules_python.bzl", "rules_python_setup")

rules_python_setup()

#==================== rules_proto ====================

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()
