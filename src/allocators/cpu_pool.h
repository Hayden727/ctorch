//===- src/allocators/cpu_pool.h - CPU pool allocator ----------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thread-local size-class pool allocator for CPU memory. Aligns every
/// allocation to a cache line so SIMD loops can vectorize. Allocations larger
/// than `kCachePoolMaxBytes` bypass the pool and call the system aligned
/// allocator directly.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_SRC_ALLOCATORS_CPU_POOL_H
#define CTORCH_SRC_ALLOCATORS_CPU_POOL_H

#include "ctorch/allocator.h"

#include <cstddef>

namespace ctorch::detail {

/// Cache-line alignment. Sized to the widest SIMD register on supported
/// architectures (AVX-512: 64 B).
inline constexpr std::size_t kCpuAlignment = 64;

/// Allocations larger than this bypass the cache and go straight to the
/// system aligned allocator. 1 MiB matches the issue spec.
inline constexpr std::size_t kCachePoolMaxBytes = 1024UL * 1024UL;

/// Number of size classes the pool tracks. `2^kCpuPoolNumSizeClasses` covers
/// allocations up to a few GiB; allocations above kCachePoolMaxBytes bypass.
inline constexpr int kCpuPoolNumSizeClasses = 32;

class CpuPoolAllocator final : public Allocator {
  public:
    CpuPoolAllocator() = default;
    ~CpuPoolAllocator() override;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* p, std::size_t bytes) override;

    /// Drops every cached block back to the system allocator. Exposed for
    /// tests and tools that want to measure underlying allocator pressure.
    void empty_cache();
};

} // namespace ctorch::detail

#endif // CTORCH_SRC_ALLOCATORS_CPU_POOL_H
