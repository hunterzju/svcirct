find_package(MLIR REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src/standalone/include)
include_directories(${PROJECT_BINARY_DIR}/src/standalone/include)

function(add_test target)
    add_executable(${target} ${target}.cpp)
    set(extra_lib ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../extern/googletest/googletest/include
            ${PRE_SLANG_DIR}/include)
    target_link_libraries(${target} PRIVATE gtest gtest_main svcirct ${extra_lib})
    gtest_discover_tests(${target})
endfunction()

add_test(test_ast)
add_test(test_mlir)
