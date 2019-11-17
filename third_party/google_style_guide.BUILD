# copied from https://github.com/ApolloAuto/apollo/blob/master/third_party/google_styleguide.BUILD

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# We can't set name="cpplint" here because that's the directory name so the
# sandbox gets confused.  We'll give it a private name with a public alias.
py_binary(
    name = "cpplint_binary",
    srcs = ["cpplint/cpplint.py"],
    imports = ["cpplint"],
    main = "cpplint/cpplint.py",
    visibility = [],
)

alias(
    name = "cpplint",
    actual = ":cpplint_binary",
)
# exports_files(["cpplint/cpplint.py"])
