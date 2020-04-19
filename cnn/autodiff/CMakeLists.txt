
add_library(autodiff INTERFACE)
target_include_directories(autodiff INTERFACE ${CMAKE_SOURCE_DIR})
target_compile_features(autodiff INTERFACE cxx_std_11)

add_executable(memory_pool_test memory_pool_test.cc)
target_link_libraries(memory_pool_test
  PRIVATE
    autodiff
    gtest
    gtest_main
)

add_test(NAME Test.memory_pool
  COMMAND $<TARGET_FILE:memory_pool_test>
)
