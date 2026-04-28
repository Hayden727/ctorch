//===- tests/bench/reference_cub_sum.h - cub::DeviceReduce::Sum ---*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Hand-written CUB reference: launches `cub::DeviceReduce::Sum` once on
/// the input tensor's device buffer and returns the elapsed wall-clock
/// seconds. Lives in its own .cu TU so the rest of the bench stays plain
/// C++ (nvcc cannot instantiate `dispatch::call<...>` cleanly).
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_TESTS_BENCH_REFERENCE_CUB_SUM_H
#define CTORCH_TESTS_BENCH_REFERENCE_CUB_SUM_H

#include "ctorch/tensor.h"

namespace ctorch::bench {

/// One-shot timed call to `cub::DeviceReduce::Sum`. `a` and `out` must
/// be 1-D fp32 CUDA tensors, `a` of length kN, `out` 0-d. Allocates a
/// temp scratch buffer once on first call (warm-up); pass `warmup=true`
/// to drop the elapsed time.
double time_reference_cub_sum(const Tensor& a, Tensor& out, bool warmup = false);

} // namespace ctorch::bench

#endif // CTORCH_TESTS_BENCH_REFERENCE_CUB_SUM_H
