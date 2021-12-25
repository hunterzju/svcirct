function(add_warning target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-error=attributes)
    endif()
endfunction()

add_library(svcirct ast.cpp)
target_link_libraries(svcirct PUBLIC slangcompiler)
# use some libraries already used for slang to avoid linking errors
if (PRE_SLANG_DIR)
    target_include_directories(svcirct PUBLIC
            ${PRE_SLANG_DIR}/include
            )
else()
    target_include_directories(svcirct PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/slang/external
    )
endif()


# # force to use old fashion of span since slang is compiled against C++17
# target_compile_definitions(svcirct PUBLIC span_CONFIG_SELECT_SPAN=1)
# add_warning(svcirct)

add_subdirectory(standalone)
