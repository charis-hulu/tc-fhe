function(add_btc_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE btc)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_btc_test(test_matmul)
add_btc_test(test_algorithms)
add_btc_test(test_transitive_closure)
