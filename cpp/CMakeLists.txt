cmake_minimum_required(VERSION 3.20)
project(tenferro_libtorch_benchmark LANGUAGES CXX)

option(BUILD_LIBTORCH_BENCHMARK "Build the LibTorch benchmark executable" ON)
option(REQUIRE_TORCH_OPENBLAS "Require libtorch_cpu to be linked against OpenBLAS" ON)
set(OPENBLAS_ROOT "" CACHE PATH "OpenBLAS installation prefix")

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

if(BUILD_LIBTORCH_BENCHMARK)
    find_package(Torch REQUIRED)

    if(REQUIRE_TORCH_OPENBLAS)
        set(TORCH_CPU_LIBRARY "")
        set(TORCH_CPU_CANDIDATES
            "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.dylib"
            "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so"
            "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.a"
        )
        foreach(candidate IN LISTS TORCH_CPU_CANDIDATES)
            if(EXISTS "${candidate}")
                set(TORCH_CPU_LIBRARY "${candidate}")
                break()
            endif()
        endforeach()
        if(NOT TORCH_CPU_LIBRARY)
            message(FATAL_ERROR
                "Could not locate libtorch_cpu for OpenBLAS linkage inspection. "
                "Set Torch_DIR or CMAKE_PREFIX_PATH to an OpenBLAS-linked LibTorch."
            )
        endif()

        if(APPLE)
            execute_process(
                COMMAND otool -L "${TORCH_CPU_LIBRARY}"
                OUTPUT_VARIABLE TORCH_CPU_LINKS
                ERROR_VARIABLE TORCH_CPU_LINK_ERRORS
                RESULT_VARIABLE TORCH_CPU_LINK_RESULT
            )
        elseif(UNIX)
            execute_process(
                COMMAND ldd "${TORCH_CPU_LIBRARY}"
                OUTPUT_VARIABLE TORCH_CPU_LINKS
                ERROR_VARIABLE TORCH_CPU_LINK_ERRORS
                RESULT_VARIABLE TORCH_CPU_LINK_RESULT
            )
        else()
            message(FATAL_ERROR
                "OpenBLAS linkage inspection is only implemented for macOS/Linux."
            )
        endif()

        if(NOT TORCH_CPU_LINK_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Failed to inspect ${TORCH_CPU_LIBRARY}: ${TORCH_CPU_LINK_ERRORS}"
            )
        endif()

        string(TOLOWER "${TORCH_CPU_LINKS}" TORCH_CPU_LINKS_LOWER)
        if(NOT TORCH_CPU_LINKS_LOWER MATCHES "openblas")
            message(FATAL_ERROR
                "Torch CPU library is not linked against OpenBLAS: ${TORCH_CPU_LIBRARY}\n"
                "Detected links:\n${TORCH_CPU_LINKS}\n"
                "Build or select a LibTorch/PyTorch installation with BLAS=OpenBLAS."
            )
        endif()

        message(STATUS "Verified OpenBLAS-linked Torch CPU library: ${TORCH_CPU_LIBRARY}")
    endif()

    find_path(OPENBLAS_INCLUDE_DIR
        NAMES cblas.h
        HINTS ${OPENBLAS_ROOT}
        PATH_SUFFIXES include include/openblas
    )
    find_library(OPENBLAS_LIBRARY
        NAMES openblas openblasp
        HINTS ${OPENBLAS_ROOT}
        PATH_SUFFIXES lib lib64
    )
    if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
        message(FATAL_ERROR
            "OpenBLAS was not found. Set -DOPENBLAS_ROOT=/path/to/openblas "
            "or provide OPENBLAS_INCLUDE_DIR and OPENBLAS_LIBRARY."
        )
    endif()

    find_package(nlohmann_json QUIET)
    if(NOT nlohmann_json_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()

    add_executable(benchmark_libtorch
        benchmark_libtorch.cpp
    )
    target_compile_definitions(benchmark_libtorch PRIVATE
        TENFERRO_BENCHMARK_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/instances"
    )
    target_link_libraries(benchmark_libtorch PRIVATE
        einsum_plan
        "${TORCH_LIBRARIES}"
        nlohmann_json::nlohmann_json
    )
endif()
