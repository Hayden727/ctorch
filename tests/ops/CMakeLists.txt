ctorch_add_test(promotion_test
  SOURCES promotion_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(broadcast_test
  SOURCES broadcast_test.cpp
  LIBS    ctorch_core
)
# broadcast_test reaches into private headers under src/ops/.
target_include_directories(broadcast_test PRIVATE ${PROJECT_SOURCE_DIR}/src)

ctorch_add_test(binary_test
  SOURCES binary_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(inplace_test
  SOURCES inplace_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(unary_test
  SOURCES unary_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(reduction_axis_test
  SOURCES reduction_axis_test.cpp
  LIBS    ctorch_core
)
# reduction_axis_test reaches into private headers under src/ops/.
target_include_directories(reduction_axis_test PRIVATE ${PROJECT_SOURCE_DIR}/src)

ctorch_add_test(reduction_value_test
  SOURCES reduction_value_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(reduction_dtype_test
  SOURCES reduction_dtype_test.cpp
  LIBS    ctorch_core
)

if(CTORCH_CUDA)
  ctorch_add_test(cuda_smoke_test
    SOURCES cuda_smoke_test.cpp
    LIBS    ctorch_core CUDA::cudart
  )
else()
  # Compile the placeholder in CPU-only builds too so it gets covered by
  # gtest_discover_tests and reports a SKIP rather than disappearing.
  ctorch_add_test(cuda_smoke_test
    SOURCES cuda_smoke_test.cpp
    LIBS    ctorch_core
  )
endif()
