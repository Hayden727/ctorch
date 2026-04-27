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
  allocators/cpu_pool.cpp
  allocators/default_allocator.cpp
)

target_include_directories(ctorch_core
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(ctorch_core PUBLIC cxx_std_20)
target_compile_options(ctorch_core PRIVATE ${CTORCH_WARNINGS})

if(CTORCH_CUDA)
  target_sources(ctorch_core PRIVATE
    cuda/stream.cpp
    allocators/cuda_caching.cpp
  )
  target_compile_definitions(ctorch_core PUBLIC CTORCH_HAS_CUDA=1)
  target_link_libraries(ctorch_core PUBLIC CUDA::cudart)
endif()
