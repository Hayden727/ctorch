if(CTORCH_CUDA)
  ctorch_add_test(add_bench
    SOURCES add_bench.cpp reference_add_kernel.cu
    LIBS    ctorch_core CUDA::cudart
  )
else()
  # CPU-only build: there's no nvcc to compile reference_add_kernel.cu, so
  # we drop it. The CUDA-specific code in add_bench.cpp is gated by
  # CTORCH_HAS_CUDA, which is only defined when CTORCH_CUDA is ON.
  ctorch_add_test(add_bench
    SOURCES add_bench.cpp
    LIBS    ctorch_core
  )
endif()
