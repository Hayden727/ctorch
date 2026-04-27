//===- src/allocators/cuda_caching.h - CUDA caching allocator --*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Per-stream caching allocator. A simplified version of PyTorch's caching
/// allocator: blocks are pooled by `(stream, size_class)`. Misses fall
/// through to `cudaMalloc`. `empty_cache()` drains every cache back to the
/// driver. A static atomic counter tracks raw `cudaMalloc` calls so unit
/// tests can assert reuse.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_SRC_ALLOCATORS_CUDA_CACHING_H
#define CTORCH_SRC_ALLOCATORS_CUDA_CACHING_H

#include "ctorch/allocator.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace ctorch::detail {

inline constexpr int kCudaPoolNumSizeClasses = 32;

class CudaCachingAllocator final : public Allocator {
  public:
    CudaCachingAllocator() = default;
    ~CudaCachingAllocator() override;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* p, std::size_t bytes) override;

    /// Allocate on a specific stream so freed blocks can be reused on the
    /// same stream without synchronization.
    void* allocate_on_stream(std::size_t bytes, cudaStream_t stream);
    void deallocate_on_stream(void* p, std::size_t bytes, cudaStream_t stream);

    /// Drain every pooled block back to `cudaFree`.
    void empty_cache();

    /// Counter incremented exactly once per real `cudaMalloc` call. Used by
    /// the caching reuse test.
    static std::int64_t cuda_malloc_count();
    static void reset_cuda_malloc_count();

  private:
    struct Key {
        cudaStream_t stream;
        int size_class;
        bool operator==(const Key& other) const noexcept {
            return stream == other.stream && size_class == other.size_class;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            auto a = reinterpret_cast<std::uintptr_t>(k.stream);
            return std::hash<std::uintptr_t>()(a) ^ (std::hash<int>()(k.size_class) << 1);
        }
    };

    std::mutex mu_;
    std::unordered_map<Key, std::vector<void*>, KeyHash> pool_;
};

} // namespace ctorch::detail

#endif // CTORCH_SRC_ALLOCATORS_CUDA_CACHING_H
