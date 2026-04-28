if(CTORCH_CUDA)
  ctorch_add_test(add_bench
    SOURCES add_bench.cpp reference_add_kernel.cu
    LIBS    ctorch_core CUDA::cudart
  )

  ctorch_add_test(sum_bench
    SOURCES sum_bench.cpp reference_cub_sum.cu
    LIBS    ctorch_core CUDA::cudart
  )
  # sum_bench dispatches reductions directly via dispatch::call<SumOp>
  # and so needs the private ReductionAxes definition under src/.
  target_include_directories(sum_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
else()
  # CPU-only build: no nvcc to compile reference_*.cu, so we drop the
  # CUDA-only TUs. The CUDA-specific code is gated by CTORCH_HAS_CUDA,
  # which is only defined when CTORCH_CUDA is ON.
  ctorch_add_test(add_bench
    SOURCES add_bench.cpp
    LIBS    ctorch_core
  )

  ctorch_add_test(sum_bench
    SOURCES sum_bench.cpp
    LIBS    ctorch_core
  )
  target_include_directories(sum_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()
