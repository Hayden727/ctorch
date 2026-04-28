# Host C++ warnings. -fopenmp-simd unlocks `#pragma omp simd` (a vectoriser
# hint we use in the contiguous element-wise paths) without pulling in the
# OpenMP runtime. gcc 7+, clang 3.7+, and Apple clang all accept it. Without
# it, gcc warns -Wunknown-pragmas and -Werror builds fail.
#
# -Wno-pass-failed silences clang's "loop not vectorized: the optimizer was
# unable to perform the requested transformation" diagnostic. The pragma is
# *advisory* — when the body has a non-vectorisable call (the transcendental
# unary kernels go through libm) we fall back to scalar, and clang's
# complaint becomes a hard error under -Werror. gcc never emits it.
set(CTORCH_WARNINGS
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unused-parameter
  -fopenmp-simd
  -Wno-pass-failed
)

if(CTORCH_WERROR)
  list(APPEND CTORCH_WARNINGS -Werror)
endif()

# nvcc parses -Werror as one of its own options expecting a value
# ("cross-execution-space-call", "all-warnings", ...). When CMake passes a
# bare host -Werror to nvcc, the next argument is consumed as that value
# (e.g. -Werror -MD turns into "nvcc fatal: Value '-MD' is not defined for
# option 'Werror'"). Forward host warnings explicitly via -Xcompiler so
# they reach the host backend without nvcc trying to interpret them.
#
# We deliberately drop -Wpedantic from the CUDA forwarding: nvcc emits
# intermediate .cudafe1.cpp files that use GCC-style line directives,
# which -Wpedantic flags as "style of line directive is a GCC extension"
# — under -Werror that breaks every .cu TU. Host C++ TUs still get
# -Wpedantic via CTORCH_WARNINGS.
#
# Both the .cu sources we ship and any future .cu code added under
# CTORCH_CUDA pick this up via $<$<COMPILE_LANGUAGE:CUDA>:${CTORCH_CUDA_WARNINGS}>.
set(CTORCH_CUDA_WARNINGS "-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter,-fopenmp-simd,-Wno-pass-failed")

if(CTORCH_WERROR)
  list(APPEND CTORCH_CUDA_WARNINGS "-Xcompiler=-Werror")
endif()
