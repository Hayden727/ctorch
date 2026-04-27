cmake_minimum_required(VERSION 3.20)
project(ctorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/ctorch_options.cmake)
include(cmake/ctorch_warnings.cmake)

if(CTORCH_CUDA)
  enable_language(CUDA)
  include(cmake/ctorch_cuda.cmake)
endif()

add_subdirectory(third_party)
add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
