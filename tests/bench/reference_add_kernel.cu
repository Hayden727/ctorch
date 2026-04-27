//===- tests/bench/reference_add_kernel.cu - hand-written CUDA add -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "reference_add_kernel.h"

#include "ctorch/tensor.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>

namespace ctorch::bench {

namespace {

__global__ void reference_add_kernel(float* __restrict__ out,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     std::int64_t n) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

} // namespace

double time_reference_add_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    const auto* ap = static_cast<const float*>(a.storage().data()) + a.offset();
    const auto* bp = static_cast<const float*>(b.storage().data()) + b.offset();
    auto* op_out = static_cast<float*>(out.storage().data()) + out.offset();
    const std::int64_t n = a.numel();
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    reference_add_kernel<<<blocks, kBlock>>>(op_out, ap, bp, n);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace ctorch::bench
