//===- tests/allocator/cpu_pool_test.cpp - CPU pool allocator -------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke + correctness tests for the CPU pool allocator. Exercises the
/// public Allocator API surface (everything we need is reachable through
/// `default_allocator(Device::cpu())`).
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"
#include "ctorch/device.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t kCacheLine = 64;

bool is_aligned(const void* p, std::size_t n) {
    return (reinterpret_cast<std::uintptr_t>(p) % n) == 0;
}

} // namespace

TEST(CpuPool, AllocateZeroReturnsNullAndDeallocateNullIsSafe) {
    auto* a = ctorch::default_allocator(ctorch::Device::cpu());
    void* p = a->allocate(0);
    EXPECT_EQ(p, nullptr);
    a->deallocate(nullptr, 0); // must not crash
}

TEST(CpuPool, AllocationsAreCacheLineAligned) {
    auto* a = ctorch::default_allocator(ctorch::Device::cpu());
    for (std::size_t bytes : {1UL, 16UL, 100UL, 4096UL, 32UL * 1024UL}) {
        void* p = a->allocate(bytes);
        ASSERT_NE(p, nullptr) << "bytes=" << bytes;
        EXPECT_TRUE(is_aligned(p, kCacheLine)) << "bytes=" << bytes;
        a->deallocate(p, bytes);
    }
}

TEST(CpuPool, ReuseInsideCacheReturnsSamePointer) {
    auto* a = ctorch::default_allocator(ctorch::Device::cpu());
    constexpr std::size_t kSize = 1024;

    void* p1 = a->allocate(kSize);
    a->deallocate(p1, kSize);

    void* p2 = a->allocate(kSize);
    EXPECT_EQ(p1, p2) << "size-class free list should hand back the freed block";
    a->deallocate(p2, kSize);
}

TEST(CpuPool, LargeAllocationsBypassCache) {
    auto* a = ctorch::default_allocator(ctorch::Device::cpu());
    constexpr std::size_t kBig = 4UL * 1024UL * 1024UL; // > 1 MiB threshold

    void* p = a->allocate(kBig);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(is_aligned(p, kCacheLine));
    a->deallocate(p, kBig);
}

TEST(CpuPool, ManyAllocationsAreReadAndWritable) {
    auto* a = ctorch::default_allocator(ctorch::Device::cpu());
    std::vector<void*> blocks;
    blocks.reserve(64);
    for (int i = 0; i < 64; ++i) {
        constexpr std::size_t kBytes = 256;
        auto* p = static_cast<unsigned char*>(a->allocate(kBytes));
        ASSERT_NE(p, nullptr);
        for (std::size_t b = 0; b < kBytes; ++b) {
            p[b] = static_cast<unsigned char>(i + b);
        }
        blocks.push_back(p);
    }
    for (void* p : blocks) {
        a->deallocate(p, 256);
    }
}

TEST(DefaultAllocator, CpuIsKindLevelSingleton) {
    auto* a = ctorch::default_allocator(ctorch::Device::cpu());
    auto* b = ctorch::default_allocator(ctorch::Device{ctorch::Device::Kind::CPU, 7});
    EXPECT_EQ(a, b) << "Device::index is irrelevant on CPU; one allocator covers it";
}
