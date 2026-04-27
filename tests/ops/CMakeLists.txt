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
