
include_directories(BEFORE ${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(gtest)

add_executable(gtest test_main.cpp)
target_link_libraries(gtest gtest-all-in-one)
