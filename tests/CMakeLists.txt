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
