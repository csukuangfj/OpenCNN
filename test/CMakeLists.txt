
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
    )
target_link_libraries(
    gtest
    gtest-all-in-one
    core
    )
