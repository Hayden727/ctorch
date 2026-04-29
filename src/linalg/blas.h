//===- src/linalg/blas.h - Device-agnostic GEMM wrapper -------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thin wrappers around `cblas_sgemm` / `cblas_dgemm` (CPU) and
/// `cublasSgemm` / `cublasDgemm` (CUDA). The CPU and CUDA front-doors
/// for `matmul` invoke these to keep the BLAS-flavour-specific call
/// site centralised; nothing else in the codebase touches cblas /
/// cuBLAS directly.
///
/// All entry points are row-major: `lda` / `ldb` / `ldc` are the row
/// strides of A / B / C in elements, and `ta` / `tb` flag whether A /
/// B should be transposed before multiplication. The CUDA path internally
/// applies the standard `C^T = B^T A^T` trick to feed cuBLAS's
/// column-major convention.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_LINALG_BLAS_H
#define CTORCH_LINALG_BLAS_H

namespace ctorch::linalg {

/// True iff the build was configured with `CTORCH_BLAS=ON`. The CPU
/// front-door checks this and raises a `ctorch::Error` if BLAS isn't
/// available.
bool cpu_blas_available();

/// `C := alpha * op(A) * op(B) + beta * C`, where `op(X) = X` if the
/// matching `t*` flag is `false`, `X^T` otherwise. A is `M×K` (after
/// op), B is `K×N` (after op), C is `M×N`. Layouts are row-major.
void gemm_cpu_f32(bool ta, bool tb, int M, int N, int K, float alpha, const float* a, int lda,
                  const float* b, int ldb, float beta, float* c, int ldc);
void gemm_cpu_f64(bool ta, bool tb, int M, int N, int K, double alpha, const double* a, int lda,
                  const double* b, int ldb, double beta, double* c, int ldc);

#if defined(CTORCH_HAS_CUDA)
/// CUDA equivalents — the implementation pins the calling thread to the
/// device of `c` via `cuda::DeviceGuard`, fetches a thread-local handle,
/// and issues the corresponding `cublas*gemm` call. The handle is
/// created lazily and lives until the thread exits.
void gemm_cuda_f32(bool ta, bool tb, int M, int N, int K, float alpha, const float* a, int lda,
                   const float* b, int ldb, float beta, float* c, int ldc, int device_index);
void gemm_cuda_f64(bool ta, bool tb, int M, int N, int K, double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta, double* c, int ldc, int device_index);
#endif

} // namespace ctorch::linalg

#endif // CTORCH_LINALG_BLAS_H
