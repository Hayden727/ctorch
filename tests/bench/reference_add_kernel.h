//===- tests/bench/reference_add_kernel.h - hand-written CUDA add -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tiny helper that runs a hand-written `__global__ add` on a 1-D fp32
/// CUDA tensor and returns the median wall-clock time. Lives in its own
/// .cu translation unit so the rest of the bench can stay plain C++.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_TESTS_BENCH_REFERENCE_ADD_KERNEL_H
#define CTORCH_TESTS_BENCH_REFERENCE_ADD_KERNEL_H

#include "ctorch/tensor.h"

namespace ctorch::bench {

/// Launches the reference kernel `out[i] = a[i] + b[i]` once on device,
/// synchronises before and after the launch, and returns the elapsed
/// wall-clock seconds. Inputs and output must all be 1-D fp32 CUDA
/// tensors of the same length, contiguous, offset 0.
double time_reference_add_cuda(const Tensor& a, const Tensor& b, Tensor& out);

} // namespace ctorch::bench

#endif // CTORCH_TESTS_BENCH_REFERENCE_ADD_KERNEL_H
