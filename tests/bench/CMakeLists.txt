if(CTORCH_CUDA)
  ctorch_add_test(add_bench
    SOURCES add_bench.cu
    LIBS    ctorch_core CUDA::cudart
  )
else()
  # CPU-only build: nvcc isn't available, so compile add_bench.cu as plain
  # C++. The CUDA-specific blocks inside the file are gated by
  # CTORCH_HAS_CUDA, which is only defined when CTORCH_CUDA is ON.
  set_source_files_properties(add_bench.cu PROPERTIES LANGUAGE CXX)
  ctorch_add_test(add_bench
    SOURCES add_bench.cu
    LIBS    ctorch_core
  )
endif()
