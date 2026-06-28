add_library(btc btc.cpp)

target_include_directories(btc PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(btc PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(btc PRIVATE /W4 /WX)
else()
    target_compile_options(btc PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
