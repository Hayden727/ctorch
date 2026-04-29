# CPU BLAS resolution.
#
# `find_package(BLAS)` discovers a system BLAS — OpenBLAS, Intel MKL, and
# Apple's Accelerate framework all work transparently here. The result is
# exposed as the `BLAS::BLAS` imported target (since CMake 3.18), which is
# what we link against from `ctorch_core`.
#
# We additionally need a `cblas.h` header for the C interface — many BLAS
# distributions bundle it but it isn't always on the default include path.
# Check the standard locations and the Accelerate framework, and surface
# the picked header through `CTORCH_CBLAS_HEADER` so source files can
# `#include` it indirectly.
#
# When `CTORCH_BLAS=OFF`, we ship a stub backend that throws a clear
# runtime error if `matmul` is invoked — useful for documentation /
# smoke builds without a BLAS dependency.

if(NOT CTORCH_BLAS)
  set(CTORCH_BLAS_VENDOR "none")
  return()
endif()

find_package(BLAS REQUIRED)

# Probe for a usable cblas.h. Accelerate vends it inside the framework
# header tree, OpenBLAS / Netlib BLAS install it next to libblas.
include(CheckIncludeFileCXX)

set(_ctorch_cblas_header "")

set(_ctorch_blas_extra_defines "")

if(APPLE)
  # Apple's Accelerate ships <Accelerate/Accelerate.h> which transitively
  # exposes cblas. Prefer it when BLAS resolves to Accelerate (the default
  # on macOS).
  check_include_file_cxx("Accelerate/Accelerate.h" CTORCH_HAS_ACCELERATE_HEADER)
  if(CTORCH_HAS_ACCELERATE_HEADER)
    set(_ctorch_cblas_header "Accelerate/Accelerate.h")
    # Pre-macOS-13.3 the legacy cblas_sgemm/dgemm are deprecated in favour
    # of an ILP64-aware interface guarded by ACCELERATE_NEW_LAPACK. Define
    # it so the symbols stay live without -Wno-deprecated-declarations
    # noise.
    list(APPEND _ctorch_blas_extra_defines "ACCELERATE_NEW_LAPACK=1")
  endif()
endif()

if(NOT _ctorch_cblas_header)
  check_include_file_cxx("cblas.h" CTORCH_HAS_CBLAS_HEADER)
  if(CTORCH_HAS_CBLAS_HEADER)
    set(_ctorch_cblas_header "cblas.h")
  endif()
endif()

if(NOT _ctorch_cblas_header)
  message(FATAL_ERROR
    "CTORCH_BLAS=ON but no usable cblas.h found. Install OpenBLAS "
    "(brew install openblas / apt-get install libopenblas-dev) or "
    "rebuild with -DCTORCH_BLAS=OFF.")
endif()

set(CTORCH_BLAS_VENDOR "${BLA_VENDOR}")
set(CTORCH_CBLAS_HEADER "${_ctorch_cblas_header}")
set(CTORCH_BLAS_EXTRA_DEFINES "${_ctorch_blas_extra_defines}")
