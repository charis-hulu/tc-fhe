add_executable(generate_bgv_batched_profiles generate_bgv_batched_profiles.cpp)
target_link_libraries(generate_bgv_batched_profiles PRIVATE btc)

if(TC_WITH_BGV_MPI)
    # SPMD (compute/full modes run under mpirun) but launched directly like
    # any other tool binary, not via ctest's mpirun wrapping -- see
    # tests/CMakeLists.txt's test_bgv_distributed_rowcol comment for why
    # ctest-launched MPI binaries need that treatment; this one is normally
    # invoked from a shell/PBS script or from the artifacts integration test
    # below via std::system(), not from ctest directly.
    add_executable(bgv_distributed_rowcol_artifacts bgv_distributed_rowcol_artifacts.cpp)
    target_link_libraries(bgv_distributed_rowcol_artifacts PRIVATE btc)
endif()
