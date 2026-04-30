//===- tests/ops/matmul_cuda_test.cpp - CUDA matmul smoke -----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// End-to-end correctness checks for the cuBLAS-backed matmul on real
/// hardware. Compiled into every build (so the suite still type-checks
/// in CPU-only mode) but the body is gated on `CTORCH_HAS_CUDA`. At
/// runtime each test cleanly skips when no CUDA device is reachable.
///
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#if defined(CTORCH_HAS_CUDA)

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/linalg.h"
#include "ctorch/tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::matmul;
using ctorch::Tensor;
using ctorch::transpose;

namespace {

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
}

template <class T> Tensor cpu_filled(std::vector<std::int64_t> shape, std::vector<T> values) {
    Tensor t(std::move(shape), std::is_same_v<T, double> ? dtype::float64 : dtype::float32,
             Device::cpu());
    auto* p = static_cast<T*>(t.storage().data());
    for (std::size_t i = 0; i < values.size(); ++i) {
        p[i] = values[i];
    }
    return t;
}

template <class T> bool close(T a, T b, double rel = 1e-4) {
    const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
    const double scale = std::max(1.0, std::abs(static_cast<double>(b)));
    return diff <= rel * scale;
}

template <class T> void expect_close_host(const Tensor& got_cuda, const Tensor& ref_cpu) {
    Tensor got_cpu = got_cuda.to(Device::cpu());
    ASSERT_EQ(got_cpu.shape(), ref_cpu.shape());
    const auto* g = static_cast<const T*>(got_cpu.storage().data());
    const auto* r = static_cast<const T*>(ref_cpu.storage().data());
    for (std::int64_t i = 0; i < ref_cpu.numel(); ++i) {
        EXPECT_TRUE(close<T>(g[i], r[i])) << "i=" << i << " got=" << g[i] << " ref=" << r[i];
    }
}

} // namespace

TEST(MatmulCuda, TwoDByTwoDMatchesHandComputed) {
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    Tensor a = cpu_filled<float>({2, 3}, {1, 2, 3, 4, 5, 6}).to(Device::cuda(0));
    Tensor b = cpu_filled<float>({3, 2}, {7, 8, 9, 10, 11, 12}).to(Device::cuda(0));
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.device().kind, Device::Kind::CUDA);
    Tensor ref = cpu_filled<float>({2, 2}, {58.0f, 64.0f, 139.0f, 154.0f});
    expect_close_host<float>(c, ref);
}

TEST(MatmulCuda, AsymmetricMNKLockstheColumnMajorTrick) {
    // Asymmetric M=5, N=11, K=7 — guards against the cuBLAS
    // column-major / row-major confusion documented in §4.4.
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr int M = 5, K = 7, N = 11;
    std::vector<float> av(M * K), bv(K * N);
    for (int i = 0; i < M * K; ++i) {
        av[i] = static_cast<float>((i * 7 + 3) % 13);
    }
    for (int i = 0; i < K * N; ++i) {
        bv[i] = static_cast<float>((i * 11 + 5) % 17);
    }
    Tensor a_cpu = cpu_filled<float>({M, K}, av);
    Tensor b_cpu = cpu_filled<float>({K, N}, bv);

    Tensor c_cpu_ref(std::vector<std::int64_t>{M, N}, dtype::float32, Device::cpu());
    auto* rp = static_cast<float*>(c_cpu_ref.storage().data());
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += av[i * K + k] * bv[k * N + j];
            }
            rp[i * N + j] = acc;
        }
    }

    Tensor c = matmul(a_cpu.to(Device::cuda(0)), b_cpu.to(Device::cuda(0)));
    expect_close_host<float>(c, c_cpu_ref);
}

TEST(MatmulCuda, MatmulOfTransposeUsesCpuRoundTripFallback) {
    // Codex P1: matmul(a, a.T()) on CUDA used to throw because a.T()
    // is non-contiguous. The materialise fallback now round-trips
    // through CPU; result must still match the dense GEMM.
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    Tensor a = cpu_filled<float>({2, 3}, {1, 2, 3, 4, 5, 6}).to(Device::cuda(0));
    Tensor c = matmul(a, transpose(a, 0, 1));
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2, 2}));
    Tensor ref = cpu_filled<float>({2, 2}, {1 + 4 + 9, 4 + 10 + 18, 4 + 10 + 18, 16 + 25 + 36});
    expect_close_host<float>(c, ref);
}

TEST(MatmulCuda, MixedFloatPromotesToFloat64) {
    // Codex P2: CPU and CUDA must accept the same dtype combinations.
    // float32 × float64 should promote to float64 on both devices.
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    Tensor a = cpu_filled<float>({2, 2}, {1, 2, 3, 4}).to(Device::cuda(0));
    Tensor b(std::vector<std::int64_t>{2, 2}, dtype::float64, Device::cpu());
    auto* bp = static_cast<double*>(b.storage().data());
    bp[0] = 1.0;
    bp[1] = 0.0;
    bp[2] = 0.0;
    bp[3] = 1.0;
    b = b.to(Device::cuda(0));
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.dtype(), dtype::float64);
    Tensor ref = cpu_filled<double>({2, 2}, {1.0, 2.0, 3.0, 4.0});
    expect_close_host<double>(c, ref);
}

#else // CTORCH_HAS_CUDA

TEST(MatmulCuda, BuildWithoutCudaSkips) { GTEST_SKIP() << "CUDA disabled at build time"; }

#endif // CTORCH_HAS_CUDA
