//===- tests/ops/cuda_smoke_test.cpp - CUDA elementwise smoke --*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke-checks that the CUDA element-wise op kernels produce results
/// matching the CPU path. Compiled only with CTORCH_HAS_CUDA defined, and
/// at runtime cleanly skipped on hosts that have the build but no actual
/// GPU device.
///
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#if defined(CTORCH_HAS_CUDA)

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/ops/elementwise.h"
#include "ctorch/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

using ctorch::add;
using ctorch::Device;
using ctorch::dtype;
using ctorch::mul;
using ctorch::relu;
using ctorch::Tensor;

namespace {

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
}

Tensor cpu_filled(std::vector<std::int64_t> shape, dtype dt,
                  std::initializer_list<float> values) {
    Tensor t(std::move(shape), dt, Device::cpu());
    auto* p = static_cast<float*>(t.storage().data());
    std::int64_t i = 0;
    for (auto v : values) {
        p[i++] = v;
    }
    return t;
}

std::vector<float> read_all_cpu(const Tensor& t) {
    Tensor c = t.contiguous();
    const auto* p = static_cast<const float*>(c.storage().data()) + c.offset();
    return std::vector<float>(p, p + c.numel());
}

} // namespace

TEST(CudaSmoke, AddF32SameShape) {
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    auto a = cpu_filled({4}, dtype::float32, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = cpu_filled({4}, dtype::float32, {10.0f, 20.0f, 30.0f, 40.0f});
    auto a_gpu = a.to(Device::cuda());
    auto b_gpu = b.to(Device::cuda());
    auto c_gpu = add(a_gpu, b_gpu);
    EXPECT_EQ(c_gpu.device().kind, Device::Kind::CUDA);
    auto c_cpu = c_gpu.to(Device::cpu());
    EXPECT_EQ(read_all_cpu(c_cpu),
              (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(CudaSmoke, MulBroadcastColumnByRow) {
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    auto a_cpu = cpu_filled({3, 1}, dtype::float32, {1.0f, 2.0f, 3.0f});
    auto b_cpu = cpu_filled({1, 4}, dtype::float32, {1.0f, 2.0f, 3.0f, 4.0f});
    auto c_gpu = mul(a_cpu.to(Device::cuda()), b_cpu.to(Device::cuda()));
    EXPECT_EQ(c_gpu.shape(), std::vector<std::int64_t>({3, 4}));
    EXPECT_EQ(read_all_cpu(c_gpu.to(Device::cpu())),
              (std::vector<float>{1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12}));
}

TEST(CudaSmoke, ReluClampsNegatives) {
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    auto x = cpu_filled({4}, dtype::float32, {-1.0f, 0.0f, 0.5f, 2.0f});
    auto y = relu(x.to(Device::cuda())).to(Device::cpu());
    EXPECT_EQ(read_all_cpu(y),
              (std::vector<float>{0.0f, 0.0f, 0.5f, 2.0f}));
}

#else

TEST(CudaSmoke, BuildWithoutCudaSkips) {
    GTEST_SKIP() << "ctorch built without CUDA support";
}

#endif // CTORCH_HAS_CUDA
