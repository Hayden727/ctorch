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
