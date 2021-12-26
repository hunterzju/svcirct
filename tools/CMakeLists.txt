find_package(MLIR REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src/standalone/include)
include_directories(${PROJECT_BINARY_DIR}/src/standalone/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRStandalone
        )
add_llvm_executable(svcirct-opt svcirct.cpp)

llvm_update_compile_flags(svcirct-opt)
target_link_libraries(svcirct-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(svcirct-opt)