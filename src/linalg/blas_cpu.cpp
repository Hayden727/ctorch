//===- src/linalg/blas_cpu.cpp - CPU GEMM wrapper -------------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// `cblas_sgemm` / `cblas_dgemm` adapter. The CPU `matmul` backend goes
/// through `gemm_cpu_f32` / `gemm_cpu_f64` so the cblas call-site is in
/// exactly one place — switching BLAS providers (Apple Accelerate,
/// OpenBLAS, MKL) is therefore a single-file change.
///
/// When `CTORCH_BLAS=OFF` the build still ships this TU but the body is
/// elided to a stub; the matmul front-door consults `cpu_blas_available`
/// and raises a clearly-worded runtime error instead.
///
//===----------------------------------------------------------------------===//

#include "linalg/blas.h"

#if defined(CTORCH_HAS_BLAS)
#include CTORCH_CBLAS_HEADER
#endif

namespace ctorch::linalg {

bool cpu_blas_available() {
#if defined(CTORCH_HAS_BLAS)
    return true;
#else
    return false;
#endif
}

#if defined(CTORCH_HAS_BLAS)

namespace {

CBLAS_TRANSPOSE to_cblas(bool t) { return t ? CblasTrans : CblasNoTrans; }

} // namespace

void gemm_cpu_f32(bool ta, bool tb, int M, int N, int K, float alpha, const float* a, int lda,
                  const float* b, int ldb, float beta, float* c, int ldc) {
    cblas_sgemm(CblasRowMajor, to_cblas(ta), to_cblas(tb), M, N, K, alpha, a, lda, b, ldb, beta, c,
                ldc);
}

void gemm_cpu_f64(bool ta, bool tb, int M, int N, int K, double alpha, const double* a, int lda,
                  const double* b, int ldb, double beta, double* c, int ldc) {
    cblas_dgemm(CblasRowMajor, to_cblas(ta), to_cblas(tb), M, N, K, alpha, a, lda, b, ldb, beta, c,
                ldc);
}

#else

// Stubs: matmul front-door checks `cpu_blas_available()` first and
// throws before any of these are reached. We still need definitions so
// the symbols resolve at link time when CTORCH_BLAS is OFF.
void gemm_cpu_f32(bool, bool, int, int, int, float, const float*, int, const float*, int, float,
                  float*, int) {}
void gemm_cpu_f64(bool, bool, int, int, int, double, const double*, int, const double*, int, double,
                  double*, int) {}

#endif // CTORCH_HAS_BLAS

} // namespace ctorch::linalg
