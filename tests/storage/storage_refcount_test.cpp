//===- tests/storage/storage_refcount_test.cpp - Storage refcount ---------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage refcount + allocator deallocation behavior. Asserts that aliasing
/// Storage handles share a buffer, that the buffer survives until the last
/// alias drops, and that the underlying allocator's `deallocate` runs exactly
/// once at end-of-life (Issue 02 §AC4).
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "ctorch/storage.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <new>

namespace {

class CountingAllocator final : public ctorch::Allocator {
  public:
    void* allocate(std::size_t bytes) override {
        ++allocate_calls;
        last_bytes = bytes;
        if (bytes == 0) {
            return nullptr;
        }
        void* p = std::malloc(bytes);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return p;
    }

    void deallocate(void* p, std::size_t bytes) override {
        ++deallocate_calls;
        last_dealloc_bytes = bytes;
        std::free(p);
    }

    int allocate_calls = 0;
    int deallocate_calls = 0;
    std::size_t last_bytes = 0;
    std::size_t last_dealloc_bytes = 0;
};

} // namespace

TEST(Storage, ZeroFillsBufferOnConstruction) {
    CountingAllocator alloc;
    ctorch::Storage s(64, ctorch::Device::cpu(), &alloc);
    auto* bytes = static_cast<unsigned char*>(s.data());
    for (std::size_t i = 0; i < s.nbytes(); ++i) {
        EXPECT_EQ(bytes[i], 0u) << "byte " << i;
    }
}

TEST(Storage, CopyAliasesBufferAndSharesRefcount) {
    CountingAllocator alloc;
    ctorch::Storage s1(128, ctorch::Device::cpu(), &alloc);
    ctorch::Storage s2 = s1;

    EXPECT_EQ(s1.data(), s2.data());
    EXPECT_EQ(s1.use_count(), 2);
    EXPECT_EQ(alloc.allocate_calls, 1);
    EXPECT_EQ(alloc.deallocate_calls, 0);
}

TEST(Storage, DeallocatesExactlyOnceAfterLastAliasDrops) {
    CountingAllocator alloc;
    {
        ctorch::Storage s1(256, ctorch::Device::cpu(), &alloc);
        {
            ctorch::Storage s2 = s1;
            EXPECT_EQ(alloc.deallocate_calls, 0);
        }
        // s2 is gone; buffer must still be valid.
        EXPECT_EQ(alloc.deallocate_calls, 0);
        EXPECT_EQ(s1.use_count(), 1);
    }
    EXPECT_EQ(alloc.allocate_calls, 1);
    EXPECT_EQ(alloc.deallocate_calls, 1);
    EXPECT_EQ(alloc.last_dealloc_bytes, 256u);
}

TEST(Storage, DeviceTagIsRoundtripped) {
    CountingAllocator alloc;
    ctorch::Storage s(8, ctorch::Device::cpu(), &alloc);
    EXPECT_TRUE(s.device().is_cpu());
    EXPECT_EQ(s.device().index, 0);
}

// When zero_fill throws (here: requesting a CUDA storage in a CPU-only build
// makes zero_fill raise), the just-allocated buffer must be returned to the
// allocator. Without the try/catch in StorageImpl's constructor the buffer
// would leak because ~StorageImpl never runs on a partially-constructed
// object.
TEST(Storage, ConstructorThatFailsZeroFillReturnsBufferToAllocator) {
#if defined(CTORCH_HAS_CUDA)
    GTEST_SKIP() << "exercise needs a build without CUDA support";
#else
    CountingAllocator alloc;
    EXPECT_THROW({ ctorch::Storage s(64, ctorch::Device::cuda(0), &alloc); }, std::runtime_error);
    EXPECT_EQ(alloc.allocate_calls, 1);
    EXPECT_EQ(alloc.deallocate_calls, 1)
        << "allocator must reclaim the buffer when zero_fill throws";
    EXPECT_EQ(alloc.last_dealloc_bytes, 64u);
#endif
}
