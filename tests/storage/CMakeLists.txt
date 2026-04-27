ctorch_add_test(storage_refcount_test
  SOURCES storage_refcount_test.cpp
  LIBS    ctorch_core
)

if(CTORCH_CUDA)
  ctorch_add_test(storage_cuda_test
    SOURCES storage_cuda_test.cpp
    LIBS    ctorch_core CUDA::cudart
  )
endif()
