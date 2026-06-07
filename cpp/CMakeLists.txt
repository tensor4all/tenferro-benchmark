cmake_minimum_required(VERSION 3.20)
project(tenferro_benchmark_cpp_helpers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(einsum_plan STATIC
    einsum_plan.cpp
)
target_include_directories(einsum_plan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

enable_testing()
add_executable(einsum_plan_test
    einsum_plan_test.cpp
)
target_link_libraries(einsum_plan_test PRIVATE einsum_plan)
add_test(NAME einsum_plan_test COMMAND einsum_plan_test)
