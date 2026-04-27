include(${PROJECT_SOURCE_DIR}/cmake/ctorch_test.cmake)

add_subdirectory(smoke)
add_subdirectory(storage)
add_subdirectory(tensor)
add_subdirectory(allocator)
add_subdirectory(dispatch)
