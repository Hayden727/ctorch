# ctorch_core: the static library every consumer (tests today, ops + bindings
# later) links against. Public include path is the project's include/ tree;
# private headers under src/ are reachable only from inside the library.

add_library(ctorch_core STATIC
  storage.cpp
  tensor.cpp
  type_promotion.cpp
  ops/broadcast.cpp
  ops/tensor_iter.cpp
  ops/cast_cpu.cpp
  ops/binary_ops_cpu.cpp
  ops/unary_ops_cpu.cpp
  ops/reduction.cpp
  ops/reduction_iter.cpp
  ops/reduction_ops_cpu.cpp
  ops/indexing_cpu.cpp
  ops/matmul.cpp
  ops/matmul_shape.cpp
  ops/matmul_cpu.cpp
  linalg/blas_cpu.cpp
  allocators/cpu_pool.cpp
  allocators/default_allocator.cpp
)

target_include_directories(ctorch_core
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(ctorch_core PUBLIC cxx_std_20)
# CXX-only flags go through CTORCH_WARNINGS verbatim. CUDA flags use the
# -Xcompiler= forwarding form so nvcc doesn't misinterpret bare -Werror.
target_compile_options(ctorch_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${CTORCH_WARNINGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CTORCH_CUDA_WARNINGS}>
)

if(CTORCH_CUDA)
  target_sources(ctorch_core PRIVATE
    cuda/stream.cpp
    allocators/cuda_caching.cpp
    ops/binary_ops_cuda.cu
    ops/unary_ops_cuda.cu
    ops/reduction_ops_cuda.cu
    ops/indexing_cuda.cu
    ops/matmul_cuda.cu
    linalg/blas_cuda.cu
  )
  target_compile_definitions(ctorch_core PUBLIC CTORCH_HAS_CUDA=1)
  target_link_libraries(ctorch_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(CTORCH_BLAS)
  target_compile_definitions(ctorch_core PRIVATE
    CTORCH_HAS_BLAS=1
    "CTORCH_CBLAS_HEADER=<${CTORCH_CBLAS_HEADER}>"
    ${CTORCH_BLAS_EXTRA_DEFINES}
  )
  target_link_libraries(ctorch_core PRIVATE BLAS::BLAS)
  if(CTORCH_CBLAS_INCLUDE_DIR)
    # cblas.h discovered outside the compiler's default search (e.g.
    # Homebrew/Apt non-system prefixes); add the resolved path so
    # blas_cpu.cpp can pick it up.
    target_include_directories(ctorch_core PRIVATE ${CTORCH_CBLAS_INCLUDE_DIR})
  endif()
endif()
