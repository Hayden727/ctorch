//===- tests/ops/reduction_alloc_test.cpp - no-heap-alloc check ---------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies issue 09 §N3: the CPU `sum` hot path performs no heap
/// allocation beyond the output tensor itself. Wraps the default CPU
/// allocator with `CountingAllocator` for the duration of the call,
/// dispatches the kernel directly (skipping the front-door's output
/// allocation so the count we measure is purely kernel overhead), and
/// asserts the count delta is zero.
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/ops/reduction.h"
#include "ctorch/tensor.h"

#include "allocators/counting_allocator.h"
#include "ops/reduction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::CountingAllocator;
using ctorch::default_allocator;
using ctorch::Device;
using ctorch::dtype;
using ctorch::set_default_allocator;
using ctorch::sum;
using ctorch::Tensor;
using ctorch::dispatch::call;

namespace {

class AllocatorOverrideGuard {
  public:
    AllocatorOverrideGuard(Device device, ctorch::Allocator* a)
        : device_(device), prev_(set_default_allocator(device, a)) {}
    ~AllocatorOverrideGuard() { set_default_allocator(device_, prev_); }
    AllocatorOverrideGuard(const AllocatorOverrideGuard&) = delete;
    AllocatorOverrideGuard& operator=(const AllocatorOverrideGuard&) = delete;

  private:
    Device device_;
    ctorch::Allocator* prev_;
};

} // namespace

TEST(ReductionSum, KernelDoesNotHeapAllocateOnHotPath) {
    // Pre-allocate input + output outside the override so those don't
    // count toward the hot-path allocation budget.
    Tensor x({1024}, dtype::float32, Device::cpu());
    Tensor out({}, dtype::float32, Device::cpu());

    auto* base = default_allocator(Device::cpu());
    CountingAllocator counter(base);
    AllocatorOverrideGuard guard(Device::cpu(), &counter);

    // Build a whole-tensor ReductionAxes manually so we can dispatch
    // the kernel directly, bypassing the front-door's output Storage
    // allocation. What remains is purely kernel overhead.
    ctorch::ops::ReductionAxes ax{};
    ax.rank = 1;
    ax.reduce[0] = true;
    ax.kept_numel = 1;
    ax.reduced_numel = x.numel();

    const auto baseline_alloc = counter.alloc_calls();
    call<ctorch::op::SumOp>(Device::Kind::CPU, x, out, ax);
    EXPECT_EQ(counter.alloc_calls() - baseline_alloc, 0u)
        << "CPU sum kernel hot path heap-allocated; expected zero allocations";
}

TEST(ReductionSum, FrontDoorAllocatesOnlyOutputStorage) {
    // The public sum() front-door allocates exactly one Tensor (the
    // output). Verify that — it's the upper bound the no-heap-alloc
    // claim implies for the user-visible API.
    Tensor x({1024}, dtype::float32, Device::cpu());

    auto* base = default_allocator(Device::cpu());
    CountingAllocator counter(base);
    AllocatorOverrideGuard guard(Device::cpu(), &counter);

    const auto baseline_alloc = counter.alloc_calls();
    auto y = sum(x);
    EXPECT_EQ(counter.alloc_calls() - baseline_alloc, 1u)
        << "ctorch::sum allocated more than the output tensor";
    EXPECT_TRUE(y.shape().empty());
}
