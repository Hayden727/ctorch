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
#include <cstdlib>
#include <new>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ctorch::detail {

namespace {

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

// Owning per-thread cache. Drains its blocks back to the system allocator
// inside the destructor body, *before* the std::array members are destroyed,
// so we never observe a destruction-order race against the array. (Earlier
// versions kept the array as a free-standing thread_local with a separate
// `ThreadDrainGuard` thread_local; on glibc that variant raced because the
// guard was constructed first and therefore destroyed last, reading already-
// destroyed std::vector storage.)
struct ThreadCache {
    std::array<std::vector<void*>, kCpuPoolNumSizeClasses> free_lists{};

    ~ThreadCache() {
        for (auto& fl : free_lists) {
            for (void* p : fl) {
                aligned_free_bytes(p);
            }
            fl.clear();
        }
    }
};

thread_local ThreadCache g_cache;

} // namespace

CpuPoolAllocator::~CpuPoolAllocator() = default;

void* CpuPoolAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    if (bytes > kCachePoolMaxBytes) {
        // Bypass: allocate the requested size rounded up to alignment and
        // never cache. posix_memalign accepts arbitrary sizes when alignment
        // is a multiple of sizeof(void*) and a power of two; we conform.
        std::size_t padded = (bytes + kCpuAlignment - 1) & ~(kCpuAlignment - 1);
        return aligned_alloc_bytes(padded);
    }

    std::size_t pow2 = round_up_pow2(bytes);
    if (pow2 < kCpuAlignment) {
        pow2 = kCpuAlignment;
    }
    int idx = size_class_index(pow2);
    auto& fl = g_cache.free_lists.at(static_cast<std::size_t>(idx));
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
    g_cache.free_lists.at(static_cast<std::size_t>(idx)).push_back(p);
}

void CpuPoolAllocator::empty_cache() {
    for (auto& fl : g_cache.free_lists) {
        for (void* p : fl) {
            aligned_free_bytes(p);
        }
        fl.clear();
    }
}

} // namespace ctorch::detail
