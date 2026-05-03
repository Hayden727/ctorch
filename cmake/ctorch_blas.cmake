# CPU BLAS resolution.
#
# `find_package(BLAS)` discovers a system BLAS — OpenBLAS, Intel MKL, and
# Apple's Accelerate framework all work transparently here. The result is
# exposed as the `BLAS::BLAS` imported target (since CMake 3.18), which is
# what we link against from `ctorch_core`.
#
# We additionally need a `cblas.h` header for the C interface — many BLAS
# distributions bundle it but it isn't always on the default include path.
# Probe Accelerate (only when the resolved vendor actually IS Accelerate),
# then fall back to `find_path` with hints covering Homebrew, Apt, and
# Cellar prefixes. Surfaced through `CTORCH_CBLAS_HEADER` and (when
# discovered outside the default search) `CTORCH_CBLAS_INCLUDE_DIR`.
#
# When `CTORCH_BLAS=OFF`, we ship a stub backend that throws a clear
# runtime error if `matmul` is invoked — useful for documentation /
# smoke builds without a BLAS dependency.

if(NOT CTORCH_BLAS)
  set(CTORCH_BLAS_VENDOR "none")
  return()
endif()

find_package(BLAS REQUIRED)

include(CheckIncludeFileCXX)

set(_ctorch_cblas_header "")
set(_ctorch_cblas_include_dir "")
set(_ctorch_blas_extra_defines "")

# Detect whether the resolved BLAS is Apple's Accelerate. This is the
# only configuration where `<Accelerate/Accelerate.h>` is the right
# header — using it against an OpenBLAS / MKL install would compile
# against Accelerate's CBLAS prototypes while linking a different ABI.
set(_ctorch_using_accelerate FALSE)
if(APPLE)
  set(_vendor_allows_accelerate TRUE)
  if(DEFINED BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "" AND
     NOT BLA_VENDOR MATCHES "^(Apple|Apple_NAS|Default|All|Generic)$")
    set(_vendor_allows_accelerate FALSE)
  endif()
  if(_vendor_allows_accelerate)
    foreach(_lib IN LISTS BLAS_LIBRARIES)
      if(_lib MATCHES "[Aa]ccelerate")
        set(_ctorch_using_accelerate TRUE)
        break()
      endif()
    endforeach()
  endif()
endif()

if(_ctorch_using_accelerate)
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
  # Probe `cblas.h` honouring the BLAS install prefix. `find_path` walks
  # the explicit HINTS first, then CMake's default search paths, so
  # Homebrew (`/opt/homebrew/opt/openblas`), Apt (`/usr/include/openblas`,
  # `/usr/include/x86_64-linux-gnu`), and `/usr/local` all resolve.
  find_path(_ctorch_cblas_path cblas.h
    HINTS
      ${BLAS_INCLUDE_DIRS}
      /opt/homebrew/opt/openblas/include
      /usr/local/opt/openblas/include
      /opt/homebrew/include
      /usr/local/include
      /usr/include/openblas
      /usr/include/x86_64-linux-gnu/openblas-pthread
      /usr/include/x86_64-linux-gnu
      /usr/include
  )
  if(_ctorch_cblas_path)
    set(_ctorch_cblas_header "cblas.h")
    set(_ctorch_cblas_include_dir "${_ctorch_cblas_path}")
  endif()
endif()

if(NOT _ctorch_cblas_header)
  message(FATAL_ERROR
    "CTORCH_BLAS=ON but no usable cblas.h found via find_package(BLAS) + "
    "find_path. Install OpenBLAS dev headers (brew install openblas / "
    "apt-get install libopenblas-dev) or rebuild with -DCTORCH_BLAS=OFF.")
endif()

set(CTORCH_BLAS_VENDOR "${BLA_VENDOR}")
set(CTORCH_CBLAS_HEADER "${_ctorch_cblas_header}")
set(CTORCH_CBLAS_INCLUDE_DIR "${_ctorch_cblas_include_dir}")
set(CTORCH_BLAS_EXTRA_DEFINES "${_ctorch_blas_extra_defines}")
