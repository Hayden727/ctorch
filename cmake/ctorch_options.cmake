option(CTORCH_CUDA     "Enable CUDA backend"        OFF)
option(CTORCH_ASAN     "Enable AddressSanitizer"    OFF)
option(CTORCH_COVERAGE "Enable coverage flags"      OFF)
option(CTORCH_WERROR   "Treat warnings as errors"   OFF)
# When ON (default), the build resolves a system BLAS via find_package(BLAS)
# and the CPU `matmul` backend dispatches into `cblas_sgemm`/`dgemm`. When
# OFF, ctorch builds without a BLAS dependency and `matmul` raises a
# clearly-worded runtime error if invoked. CI / smoke builds on machines
# without OpenBLAS / Apple Accelerate set this to OFF.
option(CTORCH_BLAS     "Link a CPU BLAS for matmul" ON)
