cmake_minimum_required(VERSION 3.20)
project(tc-fhe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TC_BUILD_TESTS      "Build tests"      ON)
option(TC_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TC_WITH_TFHE        "Enable TFHE-rs boolean CPU backend" OFF)
option(TC_WITH_TFHE_GPU    "Enable TFHE-rs GPU backend (requires CUDA)" OFF)
option(TC_WITH_DGHV        "Enable DGHV integer-based leveled FHE backend (requires GMP)" OFF)
option(TC_WITH_BGV         "Enable BGV leveled FHE backend via OpenFHE" OFF)
option(TC_WITH_BGV_BATCHED "Enable BGV SIMD-batched (packed) FHE backend via OpenFHE" OFF)

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

if(TC_WITH_DGHV)
    find_path(GMP_INCLUDE_DIR NAMES gmp.h)
    find_path(GMPXX_INCLUDE_DIR NAMES gmpxx.h)
    find_library(GMP_LIBRARY NAMES gmp libgmp)
    find_library(GMPXX_LIBRARY NAMES gmpxx libgmpxx)

    if(NOT GMP_INCLUDE_DIR OR NOT GMPXX_INCLUDE_DIR OR NOT GMP_LIBRARY OR NOT GMPXX_LIBRARY)
        message(FATAL_ERROR "TC_WITH_DGHV requires GMP with C++ bindings "
                             "(gmp.h, gmpxx.h, libgmp, libgmpxx). "
                             "On Debian/Ubuntu: apt install libgmp-dev")
    endif()

    target_include_directories(btc PUBLIC ${GMP_INCLUDE_DIR} ${GMPXX_INCLUDE_DIR})
    target_link_libraries(btc PUBLIC ${GMPXX_LIBRARY} ${GMP_LIBRARY})
    target_compile_definitions(btc PUBLIC TC_WITH_DGHV)
endif()

if(TC_WITH_BGV OR TC_WITH_BGV_BATCHED)
    find_package(OpenFHE CONFIG REQUIRED)

    # OpenFHE's installed CMake package does not expose a single "all
    # headers" include dir -- each module (core/pke/binfhe) has its own
    # subdirectory under OpenFHE_INCLUDE, plus a third-party include dir.
    # This mirrors OpenFHE's own CMakeLists.User.txt integration template.
    target_include_directories(btc PUBLIC
        ${OpenFHE_INCLUDE}
        ${OpenFHE_INCLUDE}/third-party/include
        ${OpenFHE_INCLUDE}/core
        ${OpenFHE_INCLUDE}/pke
        ${OpenFHE_INCLUDE}/binfhe)
    target_link_directories(btc PUBLIC ${OpenFHE_LIBDIR})
    target_link_libraries(btc PUBLIC ${OpenFHE_SHARED_LIBRARIES})
endif()

if(TC_WITH_BGV)
    target_compile_definitions(btc PUBLIC TC_WITH_BGV)
endif()

if(TC_WITH_BGV_BATCHED)
    target_compile_definitions(btc PUBLIC TC_WITH_BGV_BATCHED)

    # Only needed for the BGV-batched parameter-profile database (offline
    # generator + runtime lookup, see include/btc/bgv_parameter_profile.hpp
    # and tools/generate_bgv_batched_profiles.cpp) -- not required by any
    # other backend, so it is scoped to this option rather than being a
    # project-wide dependency.
    find_package(nlohmann_json CONFIG REQUIRED)
    target_link_libraries(btc PUBLIC nlohmann_json::nlohmann_json)
endif()

if(TC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_subdirectory(examples)
add_subdirectory(tools)
