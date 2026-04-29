//===- src/ops/matmul_cuda.cu - CUDA matmul backend -----------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA `matmul` backend: re-derives the per-batch GEMM plan and issues
/// one `cublasSgemm` / `cublasDgemm` call per batch step. Inputs are
/// pre-validated and contiguous.
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
void run_matmul_cuda(const Tensor& a, const Tensor& b, Tensor& out, const MatmulPlan& plan,
                    void (*gemm)(bool, bool, int, int, int, T, const T*, int, const T*, int, T, T*,
                                 int, int)) {
    const auto* a_base = static_cast<const T*>(a.storage().data()) + a.offset();
    const auto* b_base = static_cast<const T*>(b.storage().data()) + b.offset();
    auto* c_base = static_cast<T*>(out.storage().data()) + out.offset();

    const int M = static_cast<int>(plan.M);
    const int N = static_cast<int>(plan.N);
    const int K = static_cast<int>(plan.K);
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const int device_index = out.device().index;

    const std::size_t batch = plan.a_offsets.size();
    for (std::size_t s = 0; s < batch; ++s) {
        gemm(/*ta=*/false, /*tb=*/false, M, N, K, /*alpha=*/T{1}, a_base + plan.a_offsets[s], lda,
             b_base + plan.b_offsets[s], ldb, /*beta=*/T{0}, c_base + plan.c_offsets[s], ldc,
             device_index);
    }
}

} // namespace

void matmul_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    const MatmulPlan plan = plan_matmul(a, b);
    switch (a.dtype()) {
    case dtype::float32:
        run_matmul_cuda<float>(a, b, out, plan, &linalg::gemm_cuda_f32);
        break;
    case dtype::float64:
        run_matmul_cuda<double>(a, b, out, plan, &linalg::gemm_cuda_f64);
        break;
    default:
        throw DTypeError("ctorch::matmul: unexpected non-float dtype reached the CUDA kernel");
    }
}

} // namespace ops

} // namespace ctorch
