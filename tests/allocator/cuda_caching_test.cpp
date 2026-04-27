//===- tests/allocator/cuda_caching_test.cpp - CUDA caching reuse ---------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies §AC5: allocate / free / re-allocate on the same stream issues
/// exactly one real `cudaMalloc`. Built only when CTORCH_CUDA=ON; never
/// compiled by the upstream CI lane.
///
//===----------------------------------------------------------------------===//

#include "allocators/cuda_caching.h"
#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "cuda/stream.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(CudaCaching, ReusesBlockOnSameStream) {
    using ctorch::detail::CudaCachingAllocator;
    CudaCachingAllocator alloc;
    auto stream = ctorch::cuda::Stream::create();

    CudaCachingAllocator::reset_cuda_malloc_count();

    constexpr std::size_t kBytes = 1024UL * 1024UL; // 1 MiB

    void* p1 = alloc.allocate_on_stream(kBytes, stream.raw());
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(CudaCachingAllocator::cuda_malloc_count(), 1);

    alloc.deallocate_on_stream(p1, kBytes, stream.raw());

    void* p2 = alloc.allocate_on_stream(kBytes, stream.raw());
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(CudaCachingAllocator::cuda_malloc_count(), 1)
        << "second allocate on the same stream must reuse, not call cudaMalloc";
    EXPECT_EQ(p1, p2) << "freed pointer should be served back";

    alloc.deallocate_on_stream(p2, kBytes, stream.raw());
    alloc.empty_cache();
}

TEST(CudaCaching, EmptyCacheDrainsPool) {
    using ctorch::detail::CudaCachingAllocator;
    CudaCachingAllocator alloc;
    auto stream = ctorch::cuda::Stream::create();

    CudaCachingAllocator::reset_cuda_malloc_count();

    constexpr std::size_t kBytes = 4096;
    void* p = alloc.allocate_on_stream(kBytes, stream.raw());
    alloc.deallocate_on_stream(p, kBytes, stream.raw());
    alloc.empty_cache();

    void* q = alloc.allocate_on_stream(kBytes, stream.raw());
    EXPECT_EQ(CudaCachingAllocator::cuda_malloc_count(), 2)
        << "after empty_cache the next allocate must hit cudaMalloc again";
    alloc.deallocate_on_stream(q, kBytes, stream.raw());
    alloc.empty_cache();
}

// The allocator must not leak its own `cudaSetDevice` into caller state.
// Pin the calling thread to device 0, allocate via an allocator bound to
// device 0, and assert the current device is unchanged afterwards.
TEST(CudaCaching, AllocatePreservesCallerCurrentDevice) {
    using ctorch::detail::CudaCachingAllocator;

    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
        GTEST_SKIP() << "needs at least one visible CUDA device";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    CudaCachingAllocator alloc(0);
    auto stream = ctorch::cuda::Stream::create();

    void* p = alloc.allocate_on_stream(4096, stream.raw());
    ASSERT_NE(p, nullptr);

    int current = -1;
    EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 0) << "allocator must restore the caller's current device after allocation";

    alloc.deallocate_on_stream(p, 4096, stream.raw());
    alloc.empty_cache();
}

// Cross-device test: only meaningful with at least two visible GPUs.
// Pins the thread to device 0, allocates via an allocator bound to device 1,
// and confirms the thread is back on device 0.
TEST(CudaCaching, AllocateOnAnotherDeviceLeavesCallerOnOriginalDevice) {
    using ctorch::detail::CudaCachingAllocator;

    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "needs at least two visible CUDA devices";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    CudaCachingAllocator alloc(1);
    auto stream = ctorch::cuda::Stream::create();

    void* p = alloc.allocate_on_stream(4096, stream.raw());
    ASSERT_NE(p, nullptr);

    int current = -1;
    EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 0) << "thread must be back on device 0 after cross-device allocate";

    alloc.deallocate_on_stream(p, 4096, stream.raw());
    alloc.empty_cache();
}

// `default_allocator(Device)` must look at Device::index, not just kind,
// so every CUDA ordinal owns its own caching pool. Needs ≥2 visible GPUs.
TEST(DefaultAllocator, ReturnsDistinctInstancePerCudaDeviceIndex) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "needs at least two visible CUDA devices";
    }
    auto* a0 = ctorch::default_allocator(ctorch::Device::cuda(0));
    auto* a1 = ctorch::default_allocator(ctorch::Device::cuda(1));
    auto* a0_again = ctorch::default_allocator(ctorch::Device::cuda(0));
    EXPECT_NE(a0, a1);
    EXPECT_EQ(a0, a0_again);
}

// `default_allocator(Device::cuda(N))` must reject an out-of-range ordinal
// instead of constructing an unbounded number of allocator slots.
TEST(DefaultAllocator, RejectsOutOfRangeCudaIndex) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
        // No driver visible — every CUDA index is out of range; even cuda(0)
        // throws. Exercise that path.
        EXPECT_THROW((void)ctorch::default_allocator(ctorch::Device::cuda(0)), std::exception);
        return;
    }
    EXPECT_THROW((void)ctorch::default_allocator(ctorch::Device::cuda(device_count)),
                 std::out_of_range);
    EXPECT_THROW((void)ctorch::default_allocator(ctorch::Device::cuda(1 << 28)), std::out_of_range);
}
