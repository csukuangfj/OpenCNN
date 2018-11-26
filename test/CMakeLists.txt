
include_directories(BEFORE ${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(gtest)

add_executable(gtest
    test_main.cpp
    test_array.cpp
    test_io.cpp
    test_layer.cpp
    test_input_layer.cpp
    test_network.cpp
    test_array_math.cpp
    test_l2_loss_layer.cpp
    test_full_connected_layer.cpp
    test_optimizer.cpp
    test_softmax_layer.cpp
    test_log_loss_layer.cpp
    test_softmax_with_log_loss_layer.cpp
    test_jet.cpp
    test_convolution_layer.cpp
    test_relu_layer.cpp
    test_max_pooling_layer.cpp
    test_rng.cpp
    test_drop_out_layer.cpp
    test_batch_normalization_layer.cpp
    )
target_link_libraries(
    gtest
    gtest-all-in-one
    core
    )
