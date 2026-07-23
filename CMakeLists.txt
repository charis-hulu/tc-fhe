cmake_minimum_required(VERSION 3.20)
project(tc-fhe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TC_BUILD_TESTS      "Build tests"      ON)
option(TC_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TC_WITH_TFHE        "Enable TFHE-rs boolean CPU backend" OFF)
option(TC_WITH_TFHE_GPU    "Enable TFHE-rs GPU backend (requires CUDA)" OFF)

add_subdirectory(src)

if(TC_WITH_TFHE)
    set(TFHE_ROOT "/mnt/c/Users/chari/Documents/Projects/tfhe-rs/target/release"
        CACHE PATH "Path to TFHE-rs C API build output (contains libtfhe.a and tfhe.h)")

    add_library(tfhe_c_api STATIC IMPORTED)
    set_target_properties(tfhe_c_api PROPERTIES
        IMPORTED_LOCATION             "${TFHE_ROOT}/libtfhe.a"
        INTERFACE_INCLUDE_DIRECTORIES "${TFHE_ROOT}")

    find_package(Threads REQUIRED)
    target_link_libraries(btc PUBLIC tfhe_c_api Threads::Threads ${CMAKE_DL_LIBS} m)
    target_compile_definitions(btc PUBLIC TC_WITH_TFHE)
endif()

if(TC_WITH_TFHE_GPU)
    set(TFHE_ROOT "/mnt/c/Users/chari/Documents/Projects/tfhe-rs/target/release"
        CACHE PATH "Path to TFHE-rs C API build output")
    set(CUDA_ROOT "/usr/local/cuda-12.6"
        CACHE PATH "Path to CUDA 12.6 toolkit")

    if(NOT TARGET tfhe_c_api)
        add_library(tfhe_c_api STATIC IMPORTED)
        set_target_properties(tfhe_c_api PROPERTIES
            IMPORTED_LOCATION             "${TFHE_ROOT}/libtfhe.a"
            INTERFACE_INCLUDE_DIRECTORIES "${TFHE_ROOT}")
        find_package(Threads REQUIRED)
        target_link_libraries(btc PUBLIC tfhe_c_api Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()

    target_compile_definitions(btc PUBLIC TC_WITH_TFHE_GPU TC_WITH_TFHE)
    target_link_directories(btc PUBLIC "${CUDA_ROOT}/lib64")
    target_link_libraries(btc PUBLIC cudart cuda)
    target_include_directories(btc PUBLIC "${CUDA_ROOT}/include")
endif()

if(TC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_subdirectory(examples)
