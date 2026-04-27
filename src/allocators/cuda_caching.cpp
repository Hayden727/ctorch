//===- src/allocators/cuda_caching.cpp - CUDA caching impl ----------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the per-stream caching CUDA allocator.
///
//===----------------------------------------------------------------------===//

#include "allocators/cuda_caching.h"

#include <bit>
#include <stdexcept>

namespace ctorch::detail {

namespace {

std::atomic<std::int64_t> g_cuda_malloc_count{0};

std::size_t round_up_pow2(std::size_t n) noexcept {
    if (n <= 1) {
        return 1;
    }
    return std::size_t{1} << std::bit_width(n - 1);
}

int size_class_index(std::size_t pow2_bytes) noexcept {
    return static_cast<int>(std::bit_width(pow2_bytes)) - 1;
}

// Bind subsequent CUDA runtime calls on this thread to the allocator's
// device. Required before every `cudaMalloc`/`cudaFree` so a multi-GPU
// caller doesn't accidentally allocate on the current thread's device.
void set_device_or_throw(int device_index) {
    cudaError_t err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
        throw std::runtime_error("ctorch::CudaCachingAllocator: cudaSetDevice failed");
    }
}

} // namespace

CudaCachingAllocator::~CudaCachingAllocator() { empty_cache(); }

void* CudaCachingAllocator::allocate(std::size_t bytes) {
    return allocate_on_stream(bytes, cudaStreamLegacy);
}

void CudaCachingAllocator::deallocate(void* p, std::size_t bytes) {
    deallocate_on_stream(p, bytes, cudaStreamLegacy);
}

void* CudaCachingAllocator::allocate_on_stream(std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) {
        return nullptr;
    }
    std::size_t pow2 = round_up_pow2(bytes);
    int idx = size_class_index(pow2);
    Key key{stream, idx};

    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pool_.find(key);
        if (it != pool_.end() && !it->second.empty()) {
            void* p = it->second.back();
            it->second.pop_back();
            return p;
        }
    }

    set_device_or_throw(device_index_);
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, pow2);
    if (err != cudaSuccess || p == nullptr) {
        throw std::runtime_error("ctorch::CudaCachingAllocator: cudaMalloc failed");
    }
    g_cuda_malloc_count.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void CudaCachingAllocator::deallocate_on_stream(void* p, std::size_t bytes, cudaStream_t stream) {
    if (p == nullptr) {
        return;
    }
    std::size_t pow2 = round_up_pow2(bytes);
    int idx = size_class_index(pow2);
    Key key{stream, idx};

    std::lock_guard<std::mutex> lock(mu_);
    pool_[key].push_back(p);
}

void CudaCachingAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mu_);
    // Best-effort device pin. Failures here are swallowed because this
    // function is also called from the destructor; a throw would terminate.
    (void)cudaSetDevice(device_index_);
    for (auto& [key, blocks] : pool_) {
        for (void* p : blocks) {
            (void)cudaFree(p);
        }
        blocks.clear();
    }
    pool_.clear();
}

std::int64_t CudaCachingAllocator::cuda_malloc_count() {
    return g_cuda_malloc_count.load(std::memory_order_relaxed);
}

void CudaCachingAllocator::reset_cuda_malloc_count() {
    g_cuda_malloc_count.store(0, std::memory_order_relaxed);
}

} // namespace ctorch::detail
