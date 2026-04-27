set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

# Provide CUDA::cudart and CUDA::cuda_driver to C++ TUs that include
# cuda_runtime.h. enable_language(CUDA) handles the .cu side automatically;
# CUDAToolkit is needed for plain-C++ sources that link against cudart.
find_package(CUDAToolkit REQUIRED)
