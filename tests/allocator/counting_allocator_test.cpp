//===- tests/allocator/counting_allocator_test.cpp - hook + counter ------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies the per-Kind override slot in `default_allocator` and the
/// header-only `CountingAllocator` wrapper that backs Issue 09 §N3's
/// no-heap-alloc assertion.
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/tensor.h"

#include "allocators/counting_allocator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace {

class AllocatorOverrideGuard {
  public:
    AllocatorOverrideGuard(ctorch::Device device, ctorch::Allocator* a)
        : device_(device), prev_(ctorch::set_default_allocator(device, a)) {}
    ~AllocatorOverrideGuard() { ctorch::set_default_allocator(device_, prev_); }
    AllocatorOverrideGuard(const AllocatorOverrideGuard&) = delete;
    AllocatorOverrideGuard& operator=(const AllocatorOverrideGuard&) = delete;

  private:
    ctorch::Device device_;
    ctorch::Allocator* prev_;
};

} // namespace

TEST(CountingAllocator, TalliesAllocateAndDeallocate) {
    auto* base = ctorch::default_allocator(ctorch::Device::cpu());
    ctorch::CountingAllocator counter(base);

    void* p = counter.allocate(64);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(counter.alloc_calls(), 1u);
    EXPECT_EQ(counter.live_bytes(), 64);

    counter.deallocate(p, 64);
    EXPECT_EQ(counter.dealloc_calls(), 1u);
    EXPECT_EQ(counter.live_bytes(), 0);
}

TEST(CountingAllocator, ZeroByteAllocateDoesNotIncrementCounters) {
    auto* base = ctorch::default_allocator(ctorch::Device::cpu());
    ctorch::CountingAllocator counter(base);

    // Mirror CpuPool: allocate(0) returns nullptr and is not counted.
    void* p = counter.allocate(0);
    EXPECT_EQ(p, nullptr);
    EXPECT_EQ(counter.alloc_calls(), 0u);
    EXPECT_EQ(counter.live_bytes(), 0);

    counter.deallocate(nullptr, 0);
    EXPECT_EQ(counter.dealloc_calls(), 0u);
}

TEST(SetDefaultAllocator, OverlayRedirectsThenRestores) {
    auto* original = ctorch::default_allocator(ctorch::Device::cpu());
    ctorch::CountingAllocator counter(original);

    auto* prev = ctorch::set_default_allocator(ctorch::Device::cpu(), &counter);
    // First call returns the previous (nullptr if no other overlay was
    // installed), so the overlay slot is otherwise untouched by the test
    // suite.
    EXPECT_EQ(prev, nullptr);
    EXPECT_EQ(ctorch::default_allocator(ctorch::Device::cpu()), &counter);

    auto* after_restore = ctorch::set_default_allocator(ctorch::Device::cpu(), nullptr);
    EXPECT_EQ(after_restore, &counter);
    EXPECT_EQ(ctorch::default_allocator(ctorch::Device::cpu()), original);
}

TEST(SetDefaultAllocator, CountsTensorBackedAllocation) {
    auto* base = ctorch::default_allocator(ctorch::Device::cpu());
    ctorch::CountingAllocator counter(base);

    AllocatorOverrideGuard guard(ctorch::Device::cpu(), &counter);

    const auto baseline_alloc = counter.alloc_calls();
    const auto baseline_dealloc = counter.dealloc_calls();
    {
        ctorch::Tensor t({1024}, ctorch::dtype::float32, ctorch::Device::cpu());
        EXPECT_EQ(counter.alloc_calls() - baseline_alloc, 1u);
        EXPECT_GE(counter.live_bytes(), static_cast<std::int64_t>(1024 * sizeof(float)));
    }
    EXPECT_EQ(counter.dealloc_calls() - baseline_dealloc, 1u);
    EXPECT_EQ(counter.live_bytes(), 0);
}

TEST(CountingAllocator, ConcurrentAllocateIsRaceFree) {
    auto* base = ctorch::default_allocator(ctorch::Device::cpu());
    ctorch::CountingAllocator counter(base);

    constexpr int kThreads = 8;
    constexpr int kAllocsPerThread = 256;
    constexpr std::size_t kBytes = 32;

    std::vector<std::thread> ts;
    ts.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        ts.emplace_back([&] {
            std::vector<void*> ptrs;
            ptrs.reserve(kAllocsPerThread);
            for (int j = 0; j < kAllocsPerThread; ++j) {
                ptrs.push_back(counter.allocate(kBytes));
            }
            for (void* p : ptrs) {
                counter.deallocate(p, kBytes);
            }
        });
    }
    for (auto& t : ts) {
        t.join();
    }
    EXPECT_EQ(counter.alloc_calls(), static_cast<std::size_t>(kThreads * kAllocsPerThread));
    EXPECT_EQ(counter.dealloc_calls(), static_cast<std::size_t>(kThreads * kAllocsPerThread));
    EXPECT_EQ(counter.live_bytes(), 0);
}
