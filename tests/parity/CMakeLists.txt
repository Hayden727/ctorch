# Static helper carrying the .npy loader; both parity tests link it in
# rather than re-compiling the loader twice.
add_library(ctorch_parity_support STATIC load_npy.cpp)
target_include_directories(ctorch_parity_support
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(ctorch_parity_support PUBLIC ctorch_core)

set(CTORCH_PARITY_FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

ctorch_add_test(binary_parity_test
  SOURCES binary_parity_test.cpp
  LIBS    ctorch_parity_support
)
target_compile_definitions(binary_parity_test
  PRIVATE CTORCH_PARITY_FIXTURES_DIR="${CTORCH_PARITY_FIXTURES_DIR}"
)

ctorch_add_test(unary_parity_test
  SOURCES unary_parity_test.cpp
  LIBS    ctorch_parity_support
)
target_compile_definitions(unary_parity_test
  PRIVATE CTORCH_PARITY_FIXTURES_DIR="${CTORCH_PARITY_FIXTURES_DIR}"
)

ctorch_add_test(reduction_parity_test
  SOURCES reduction_parity_test.cpp
  LIBS    ctorch_parity_support
)
target_compile_definitions(reduction_parity_test
  PRIVATE CTORCH_PARITY_FIXTURES_DIR="${CTORCH_PARITY_FIXTURES_DIR}"
)

ctorch_add_test(indexing_parity_test
  SOURCES indexing_parity_test.cpp
  LIBS    ctorch_parity_support
)
target_compile_definitions(indexing_parity_test
  PRIVATE CTORCH_PARITY_FIXTURES_DIR="${CTORCH_PARITY_FIXTURES_DIR}"
)
