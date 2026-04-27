ctorch_add_test(cpu_pool_test
  SOURCES cpu_pool_test.cpp
  LIBS    ctorch_core
)

if(CTORCH_CUDA)
  ctorch_add_test(cuda_caching_test
    SOURCES cuda_caching_test.cpp
    LIBS    ctorch_core CUDA::cudart
  )
  # cuda_caching_test reaches into the internal allocator headers under src/.
  target_include_directories(cuda_caching_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()
