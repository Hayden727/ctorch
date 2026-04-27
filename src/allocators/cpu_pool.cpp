//===- src/allocators/cpu_pool.cpp - CPU pool implementation --------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the per-thread size-class pool allocator. Each thread
/// keeps an array of free lists indexed by `bit_width(round_up_pow2(bytes))`,
/// so deallocations on one thread never contend with another. On a cache
/// hit we just pop from the head of the relevant free list.
///
//===----------------------------------------------------------------------===//

#include "allocators/cpu_pool.h"

#include <array>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <new>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ctorch::detail {

namespace {

using FreeList = std::vector<void*>;
using FreeLists = std::array<FreeList, kCpuPoolNumSizeClasses>;

// Per-thread free lists. Holding a separate cache per thread gives lock-free
// allocate/deallocate at the cost of cross-thread reuse, which is acceptable
// for tensor workloads where each thread typically owns its own tensors.
thread_local FreeLists g_free_lists{};

// Track every block this thread has handed back to the system so destroying
// the thread (or calling empty_cache) can release them.
thread_local bool g_drain_registered = false;

std::size_t round_up_pow2(std::size_t n) noexcept {
    if (n <= 1) {
        return 1;
    }
    return std::size_t{1} << std::bit_width(n - 1);
}

int size_class_index(std::size_t pow2_bytes) noexcept {
    // bit_width(1) == 1, bit_width(2) == 2, ... index = bit_width - 1.
    return static_cast<int>(std::bit_width(pow2_bytes)) - 1;
}

void* aligned_alloc_bytes(std::size_t bytes) {
#if defined(_WIN32)
    void* p = _aligned_malloc(bytes, kCpuAlignment);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
#else
    void* p = nullptr;
    int err = posix_memalign(&p, kCpuAlignment, bytes);
    if (err != 0 || p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
#endif
}

void aligned_free_bytes(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void drain_free_lists(FreeLists& lists) noexcept {
    for (auto& fl : lists) {
        for (void* p : fl) {
            aligned_free_bytes(p);
        }
        fl.clear();
    }
}

// Per-thread sentinel that drains the thread's free lists when the thread
// exits. Constructed lazily on first allocate call.
struct ThreadDrainGuard {
    ~ThreadDrainGuard() { drain_free_lists(g_free_lists); }
};

void ensure_drain_registered() {
    if (!g_drain_registered) {
        thread_local ThreadDrainGuard guard;
        (void)guard;
        g_drain_registered = true;
    }
}

} // namespace

CpuPoolAllocator::~CpuPoolAllocator() {
    // Drain the *calling* thread's lists. Other threads' lists drain on their
    // own ThreadDrainGuard destruction. In practice CpuPoolAllocator is a
    // process-wide singleton so this destructor only runs at shutdown.
    drain_free_lists(g_free_lists);
}

void* CpuPoolAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    if (bytes > kCachePoolMaxBytes) {
        // Bypass: allocate the requested rounded-up-to-alignment size and
        // never cache. posix_memalign requires the size to be a multiple of
        // sizeof(void*); rounding up to the alignment satisfies that.
        std::size_t padded = (bytes + kCpuAlignment - 1) & ~(kCpuAlignment - 1);
        return aligned_alloc_bytes(padded);
    }

    ensure_drain_registered();

    std::size_t pow2 = round_up_pow2(bytes);
    if (pow2 < kCpuAlignment) {
        pow2 = kCpuAlignment;
    }
    int idx = size_class_index(pow2);
    auto& fl = g_free_lists.at(static_cast<std::size_t>(idx));
    if (!fl.empty()) {
        void* p = fl.back();
        fl.pop_back();
        return p;
    }
    return aligned_alloc_bytes(pow2);
}

void CpuPoolAllocator::deallocate(void* p, std::size_t bytes) {
    if (p == nullptr) {
        return;
    }
    if (bytes > kCachePoolMaxBytes) {
        aligned_free_bytes(p);
        return;
    }

    std::size_t pow2 = round_up_pow2(bytes);
    if (pow2 < kCpuAlignment) {
        pow2 = kCpuAlignment;
    }
    int idx = size_class_index(pow2);
    g_free_lists.at(static_cast<std::size_t>(idx)).push_back(p);
}

void CpuPoolAllocator::empty_cache() { drain_free_lists(g_free_lists); }

} // namespace ctorch::detail
