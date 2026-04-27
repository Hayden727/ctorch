//===- tests/bench/add_bench.cpp - Microbenchmark: add ---------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Microbenchmark for the element-wise `add` op. On CPU it just times the
/// dispatched call vs a hand-rolled tight loop. On CUDA (when CTORCH_HAS_CUDA
/// is defined and a GPU is present) it adds a hand-written reference kernel
/// and asserts the dispatched call lands within 1.10× of it on a 1<<20 fp32
/// tensor — the N2 acceptance criterion from Issue 03 §2.2.
///
/// Output is one CSV line per (op, backend, n) tuple to stdout. Designed to
/// be runnable as a normal test (gtest_discover_tests picks it up) or
/// invoked manually for ad-hoc profiling.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/ops/elementwise.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#if defined(CTORCH_HAS_CUDA)
#include "reference_add_kernel.h"
#include <cuda_runtime.h>
#endif

using ctorch::add;
using ctorch::Device;
using ctorch::dtype;
using ctorch::Tensor;

namespace {

constexpr std::int64_t kN = 1 << 20; // 1M elements
constexpr int kTrials = 25;

double median(std::vector<double>& xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

void emit_csv(const char* op, const char* backend, std::int64_t n, double seconds) {
    const double bytes = 3.0 * n * static_cast<double>(sizeof(float));
    const double gbps = (bytes / seconds) * 1e-9;
    std::cout << "csv," << op << "," << backend << "," << n << ","
              << seconds << "," << gbps << "\n";
}

#if defined(CTORCH_HAS_CUDA)

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
}

// The hand-written reference kernel and its launch wrapper live in a
// dedicated .cu translation unit so this file stays plain C++ — that way
// nvcc never tries to instantiate `dispatch::call<...>` (which it
// mishandles as an "incomplete type"). See reference_add_kernel.h.

// Bypass the public `add()` front-door so the dispatch path writes into a
// preallocated `out` instead of allocating + zero-filling a fresh tensor on
// every trial. That is what N2 (Issue 03 §2.2) actually wants to measure:
// dispatcher + indexer + kernel overhead, not Storage allocation + cudaMemset.
double time_dispatch_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    ctorch::dispatch::call<ctorch::op::AddOp>(Device::Kind::CUDA, a, b, out);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

#endif // CTORCH_HAS_CUDA

// Same shape as the CUDA case: dispatch into a preallocated output so the
// timing reflects only the dispatcher + indexer + kernel.
double time_dispatch_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    auto t0 = std::chrono::steady_clock::now();
    ctorch::dispatch::call<ctorch::op::AddOp>(Device::Kind::CPU, a, b, out);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

double time_reference_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    const auto* ap = static_cast<const float*>(a.storage().data()) + a.offset();
    const auto* bp = static_cast<const float*>(b.storage().data()) + b.offset();
    auto* op_out = static_cast<float*>(out.storage().data()) + out.offset();
    const std::int64_t n = a.numel();
    auto t0 = std::chrono::steady_clock::now();
    #pragma omp simd
    for (std::int64_t i = 0; i < n; ++i) {
        op_out[i] = ap[i] + bp[i];
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

TEST(AddBench, CpuDispatchVsReference) {
    Tensor a({kN}, dtype::float32, Device::cpu());
    Tensor b({kN}, dtype::float32, Device::cpu());
    Tensor out_disp({kN}, dtype::float32, Device::cpu());
    Tensor out_ref({kN}, dtype::float32, Device::cpu());

    // The registrar that wires `op::AddOp` into the dispatch table lives in
    // the same TU as `ctorch::add()`; calling it here forces the static
    // library linker to pull that TU in.
    (void)add(a, b);

    // Warmup.
    (void)time_dispatch_cpu(a, b, out_disp);
    (void)time_reference_cpu(a, b, out_ref);

    std::vector<double> ts_disp(kTrials);
    std::vector<double> ts_ref(kTrials);
    for (int i = 0; i < kTrials; ++i) {
        ts_disp[i] = time_dispatch_cpu(a, b, out_disp);
        ts_ref[i] = time_reference_cpu(a, b, out_ref);
    }
    const double t_disp = median(ts_disp);
    const double t_ref = median(ts_ref);
    emit_csv("add", "cpu_dispatch", kN, t_disp);
    emit_csv("add", "cpu_reference", kN, t_ref);
    SUCCEED() << "dispatch median=" << t_disp << "s ref median=" << t_ref << "s";
}

#if defined(CTORCH_HAS_CUDA)

TEST(AddBench, CudaDispatchWithin10PercentOfReference) {
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    auto a_cpu = Tensor({kN}, dtype::float32, Device::cpu());
    auto b_cpu = Tensor({kN}, dtype::float32, Device::cpu());
    auto a = a_cpu.to(Device::cuda());
    auto b = b_cpu.to(Device::cuda());
    Tensor out_disp({kN}, dtype::float32, Device::cuda());
    Tensor out_ref({kN}, dtype::float32, Device::cuda());

    // Force the CUDA-side registrar TU to be linked (see CPU comment).
    (void)add(a, b);

    // Warmup.
    (void)time_dispatch_cuda(a, b, out_disp);
    (void)ctorch::bench::time_reference_add_cuda(a, b, out_ref);

    std::vector<double> ts_disp(kTrials);
    std::vector<double> ts_ref(kTrials);
    for (int i = 0; i < kTrials; ++i) {
        ts_disp[i] = time_dispatch_cuda(a, b, out_disp);
        ts_ref[i] = ctorch::bench::time_reference_add_cuda(a, b, out_ref);
    }
    const double t_disp = median(ts_disp);
    const double t_ref = median(ts_ref);
    emit_csv("add", "cuda_dispatch", kN, t_disp);
    emit_csv("add", "cuda_reference", kN, t_ref);
    EXPECT_LE(t_disp, 1.10 * t_ref)
        << "dispatch median=" << t_disp << "s reference median=" << t_ref << "s";
}

#endif // CTORCH_HAS_CUDA
