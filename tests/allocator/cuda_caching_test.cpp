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
#include "cuda/stream.h"

#include <gtest/gtest.h>

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
