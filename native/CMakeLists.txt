cmake_minimum_required(VERSION 3.20)
project(mathfield_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(mathfield_kernels SHARED
    src/mathfield_kernels.cpp
)

target_link_libraries(mathfield_kernels PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(mathfield_kernels PRIVATE /O2 /fp:precise /W4 /MD)
else()
    target_compile_options(mathfield_kernels PRIVATE -O3 -Wall -Wextra -fPIC)
endif()

if(WIN32)
    set_target_properties(mathfield_kernels PROPERTIES OUTPUT_NAME "mathfield_kernels" PREFIX "")
else()
    # POSIX shared libraries conventionally use the lib prefix, matching the loader.
    set_target_properties(mathfield_kernels PROPERTIES OUTPUT_NAME "mathfield_kernels" PREFIX "lib")
endif()

set_target_properties(mathfield_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../mathfield/core/engine"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../mathfield/core/engine"
)
