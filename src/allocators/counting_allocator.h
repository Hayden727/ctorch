//===- src/allocators/counting_allocator.h - Instrumented allocator -------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Header-only Allocator wrapper that tallies the number of allocate /
/// deallocate calls and live byte count. Intended as the
/// "sampling allocator hook" backing the no-heap-alloc assertions in
/// Issue 09 §N3 and as a general instrumentation primitive for tests.
///
/// Counters are atomic so the wrapper is safe to share across threads.
/// The wrapped base allocator is borrowed; CountingAllocator does not
/// take ownership and the base must outlive the wrapper.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_ALLOCATORS_COUNTING_ALLOCATOR_H
#define CTORCH_ALLOCATORS_COUNTING_ALLOCATOR_H

#include "ctorch/allocator.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ctorch {

class CountingAllocator final : public Allocator {
  public:
    explicit CountingAllocator(Allocator* base) : base_(base) {}

    void* allocate(std::size_t bytes) override {
        void* p = base_->allocate(bytes);
        // Mirror the CPU pool's "0-byte allocate returns nullptr" contract
        // so this wrapper does not invent allocations the base did not do.
        if (p != nullptr) {
            alloc_calls_.fetch_add(1, std::memory_order_relaxed);
            live_bytes_.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
        }
        return p;
    }

    void deallocate(void* p, std::size_t bytes) override {
        base_->deallocate(p, bytes);
        if (p != nullptr) {
            dealloc_calls_.fetch_add(1, std::memory_order_relaxed);
            live_bytes_.fetch_sub(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
        }
    }

    std::size_t alloc_calls() const noexcept {
        return alloc_calls_.load(std::memory_order_relaxed);
    }
    std::size_t dealloc_calls() const noexcept {
        return dealloc_calls_.load(std::memory_order_relaxed);
    }
    /// Net live bytes (alloc bytes minus dealloc bytes). Signed so a buggy
    /// caller producing a deficit shows up as a negative number rather
    /// than a giant unsigned wraparound.
    std::int64_t live_bytes() const noexcept { return live_bytes_.load(std::memory_order_relaxed); }

  private:
    Allocator* base_;
    std::atomic<std::size_t> alloc_calls_{0};
    std::atomic<std::size_t> dealloc_calls_{0};
    std::atomic<std::int64_t> live_bytes_{0};
};

} // namespace ctorch

#endif // CTORCH_ALLOCATORS_COUNTING_ALLOCATOR_H
