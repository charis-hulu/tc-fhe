add_executable(bench_closure bench_closure.cpp)
target_link_libraries(bench_closure PRIVATE btc)

add_executable(gen_graphs gen_graphs.cpp)
target_link_libraries(gen_graphs PRIVATE btc)

add_executable(bench_tfhe bench_tfhe.cpp)
target_link_libraries(bench_tfhe PRIVATE btc)

add_executable(bench_tfhe_gpu bench_tfhe_gpu.cpp)
target_link_libraries(bench_tfhe_gpu PRIVATE btc)

add_executable(bench_dghv bench_dghv.cpp)
target_link_libraries(bench_dghv PRIVATE btc)

add_executable(bench_bgv bench_bgv.cpp)
target_link_libraries(bench_bgv PRIVATE btc)

add_executable(bench_bgv_batched bench_bgv_batched.cpp)
target_link_libraries(bench_bgv_batched PRIVATE btc)

add_executable(bench_bgv_rowcol bench_bgv_rowcol.cpp)
target_link_libraries(bench_bgv_rowcol PRIVATE btc)
