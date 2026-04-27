//===- src/allocators/default_allocator.cpp - Allocator registry ----------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Process-wide default allocator lookup. The CPU side hands back a single
/// pool allocator; the CUDA side maintains one caching allocator per device
/// ordinal so multi-GPU tensors get memory on the device they claim to live
/// on (Issue 02 §F7 — "pluggable per device").
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"

#include "allocators/cpu_pool.h"

#include <stdexcept>

#if defined(CTORCH_HAS_CUDA)
#include "allocators/cuda_caching.h"

#include <memory>
#include <mutex>
#include <vector>
#endif

namespace ctorch {

#if defined(CTORCH_HAS_CUDA)
namespace {

/// Returns the caching allocator that owns CUDA device \p index. Allocators
/// are constructed on demand and never destroyed; the table itself is a
/// function-local static so it stays alive until program termination.
detail::CudaCachingAllocator* cuda_allocator_for(int index) {
    if (index < 0) {
        throw std::invalid_argument("ctorch::default_allocator: negative CUDA device index");
    }
    static std::mutex mu;
    static std::vector<std::unique_ptr<detail::CudaCachingAllocator>> table;
    std::lock_guard<std::mutex> lock(mu);
    while (static_cast<int>(table.size()) <= index) {
        int next = static_cast<int>(table.size());
        table.emplace_back(std::make_unique<detail::CudaCachingAllocator>(next));
    }
    return table[static_cast<std::size_t>(index)].get();
}

} // namespace
#endif // CTORCH_HAS_CUDA

Allocator* default_allocator(Device device) {
    switch (device.kind) {
    case Device::Kind::CPU: {
        static detail::CpuPoolAllocator cpu;
        return &cpu;
    }
    case Device::Kind::CUDA: {
#if defined(CTORCH_HAS_CUDA)
        return cuda_allocator_for(device.index);
#else
        throw std::runtime_error(
            "ctorch::default_allocator: CUDA backend not built (CTORCH_CUDA=OFF)");
#endif
    }
    }
    throw std::invalid_argument("ctorch::default_allocator: unknown Device::Kind");
}

} // namespace ctorch
