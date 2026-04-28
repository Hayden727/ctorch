//===- tests/bench/reference_cub_sum.cu - cub::DeviceReduce::Sum ----------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "reference_cub_sum.h"

#include "ctorch/tensor.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace ctorch::bench {

double time_reference_cub_sum(const Tensor& a, Tensor& out, bool warmup) {
    const auto* d_in = static_cast<const float*>(a.storage().data()) + a.offset();
    auto* d_out = static_cast<float*>(out.storage().data()) + out.offset();
    const std::int64_t n = a.numel();

    // Pass 1: query temp storage size.
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, static_cast<int>(n));
    cudaMalloc(&d_temp, temp_bytes);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, static_cast<int>(n));
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    cudaFree(d_temp);
    if (warmup) {
        return 0.0;
    }
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace ctorch::bench
