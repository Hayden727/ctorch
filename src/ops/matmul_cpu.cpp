//===- src/ops/matmul_cpu.cpp - CPU matmul backend ------------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU `matmul` backend: re-derives the per-batch GEMM plan and issues
/// one `cblas_sgemm` / `cblas_dgemm` call per batch step. The shape
/// helper has already validated inner-dim compatibility — this TU just
/// has to feed BLAS the right pointers.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "linalg/blas.h"
#include "ops/matmul_shape.h"

#include <cstdint>

namespace ctorch {

namespace ops {

namespace {

template <class T>
void run_matmul_cpu(const Tensor& a, const Tensor& b, Tensor& out, const ops::MatmulPlan& plan,
                   void (*gemm)(bool, bool, int, int, int, T, const T*, int, const T*, int, T, T*,
                                int)) {
    const auto* a_base = static_cast<const T*>(a.storage().data()) + a.offset();
    const auto* b_base = static_cast<const T*>(b.storage().data()) + b.offset();
    auto* c_base = static_cast<T*>(out.storage().data()) + out.offset();

    const int M = static_cast<int>(plan.M);
    const int N = static_cast<int>(plan.N);
    const int K = static_cast<int>(plan.K);
    const int lda = K; // contiguous, no transpose: row stride of A (M×K)
    const int ldb = N; // contiguous, no transpose: row stride of B (K×N)
    const int ldc = N;

    const std::size_t batch = plan.a_offsets.size();
    for (std::size_t s = 0; s < batch; ++s) {
        gemm(/*ta=*/false, /*tb=*/false, M, N, K, /*alpha=*/T{1}, a_base + plan.a_offsets[s], lda,
             b_base + plan.b_offsets[s], ldb, /*beta=*/T{0}, c_base + plan.c_offsets[s], ldc);
    }
}

} // namespace

void matmul_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    if (!linalg::cpu_blas_available()) {
        throw Error("ctorch::matmul: this build was configured with CTORCH_BLAS=OFF; "
                    "rebuild with -DCTORCH_BLAS=ON (and a system BLAS) to enable matmul");
    }
    const MatmulPlan plan = plan_matmul(a, b);
    switch (a.dtype()) {
    case dtype::float32:
        run_matmul_cpu<float>(a, b, out, plan, &linalg::gemm_cpu_f32);
        break;
    case dtype::float64:
        run_matmul_cpu<double>(a, b, out, plan, &linalg::gemm_cpu_f64);
        break;
    default:
        // Front-door already gated this — reaching here is a programming
        // error, not user input.
        throw DTypeError("ctorch::matmul: unexpected non-float dtype reached the CPU kernel");
    }
}

} // namespace ops

} // namespace ctorch
