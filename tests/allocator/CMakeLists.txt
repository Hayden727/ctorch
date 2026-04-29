ctorch_add_test(cpu_pool_test
  SOURCES cpu_pool_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(counting_allocator_test
  SOURCES counting_allocator_test.cpp
  LIBS    ctorch_core
)
# counting_allocator_test reaches into the header-only wrapper under src/.
target_include_directories(counting_allocator_test PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(CTORCH_CUDA)
  ctorch_add_test(cuda_caching_test
    SOURCES cuda_caching_test.cpp
    LIBS    ctorch_core CUDA::cudart
  )
  # cuda_caching_test reaches into the internal allocator headers under src/.
  target_include_directories(cuda_caching_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()
