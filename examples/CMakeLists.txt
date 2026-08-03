add_executable(tfhe_closure tfhe_closure.cpp)
target_link_libraries(tfhe_closure PRIVATE btc)

add_executable(dghv_closure dghv_closure.cpp)
target_link_libraries(dghv_closure PRIVATE btc)

add_executable(bgv_closure bgv_closure.cpp)
target_link_libraries(bgv_closure PRIVATE btc)

add_executable(bgv_batched_closure bgv_batched_closure.cpp)
target_link_libraries(bgv_batched_closure PRIVATE btc)
