add_executable(bench_closure bench_closure.cpp)
target_link_libraries(bench_closure PRIVATE btc)

add_executable(bench_tfhe bench_tfhe.cpp)
target_link_libraries(bench_tfhe PRIVATE btc)
