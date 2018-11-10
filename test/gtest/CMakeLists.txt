
add_library(gtest-all-in-one gtest-all.cpp)
target_compile_definitions(
    gtest-all-in-one
    PUBLIC
    -DGTEST_HAS_PTHREAD=0
    -DGTEST_USE_OWN_TR1_TUPLE=1
    )

