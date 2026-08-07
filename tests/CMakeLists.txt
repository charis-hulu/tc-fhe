function(add_btc_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE btc)
    # WORKING_DIRECTORY = repo root (not the build dir, ctest's default) so
    # tests that load relative paths like "data/graph_N4.txt" or
    # "config/bgv_batched_profiles.json" behave the same whether run via
    # ctest or by invoking the built binary directly from the repo root.
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

add_btc_test(test_matmul)
add_btc_test(test_algorithms)
add_btc_test(test_transitive_closure)
add_btc_test(test_dghv)
add_btc_test(test_bgv)
add_btc_test(test_bgv_batched_rotation)
add_btc_test(test_bgv_batched_packing)
add_btc_test(test_bgv_batched_matmul)
add_btc_test(test_bgv_batched_tc)
add_btc_test(test_bgv_parameter_profile)
add_btc_test(test_bgv_rowcol_matmul)
add_btc_test(test_bgv_rowcol_tc)

if(TC_WITH_BGV_MPI)
    # NOT add_btc_test: this binary is SPMD and must be launched with
    # mpirun -n {1,2,4}, not invoked directly the way ctest's default
    # add_test(COMMAND ...) would -- see test_bgv_distributed_rowcol.cpp's
    # header comment for the exact mpirun invocations. Registered as three
    # separate ctest cases (one per rank count) so `ctest` still exercises
    # all three, each via its own mpirun wrapper.
    add_executable(test_bgv_distributed_rowcol test_bgv_distributed_rowcol.cpp)
    target_link_libraries(test_bgv_distributed_rowcol PRIVATE btc)

    find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun)
    if(MPIEXEC_EXECUTABLE)
        foreach(nranks 1 2 4)
            add_test(NAME test_bgv_distributed_rowcol_np${nranks}
                     COMMAND ${MPIEXEC_EXECUTABLE} -n ${nranks} $<TARGET_FILE:test_bgv_distributed_rowcol>
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endforeach()
    endif()

    # Unlike test_bgv_distributed_rowcol above, THIS binary is single-process
    # (add_btc_test-compatible) -- it shells out to the
    # bgv_distributed_rowcol_artifacts tool (itself SPMD) via std::system()
    # for its compute stage, so ctest invokes it directly like any other
    # test, not through mpirun. TOOL_BINARY_PATH is the tool's exact built
    # path, injected via $<TARGET_FILE:...> instead of guessing the build
    # directory's name -- requires tools/ to already be processed (see the
    # top-level CMakeLists.txt's add_subdirectory ordering comment).
    add_executable(test_bgv_distributed_rowcol_artifacts test_bgv_distributed_rowcol_artifacts.cpp)
    target_link_libraries(test_bgv_distributed_rowcol_artifacts PRIVATE btc)
    target_compile_definitions(test_bgv_distributed_rowcol_artifacts PRIVATE
        TOOL_BINARY_PATH="$<TARGET_FILE:bgv_distributed_rowcol_artifacts>")
    add_dependencies(test_bgv_distributed_rowcol_artifacts bgv_distributed_rowcol_artifacts)
    add_test(NAME test_bgv_distributed_rowcol_artifacts
             COMMAND test_bgv_distributed_rowcol_artifacts
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
