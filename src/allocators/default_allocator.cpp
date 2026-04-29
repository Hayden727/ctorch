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

#include <atomic>
#include <stdexcept>

#if defined(CTORCH_HAS_CUDA)
#include "allocators/cuda_caching.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#endif

namespace ctorch {

namespace {

/// Per-Kind override slot. Loaded with relaxed semantics on every
/// `default_allocator` call; when no override is installed the value is
/// `nullptr` and the call falls through to the built-in pool. The slot
/// is keyed only by `Device::Kind` because tests overwhelmingly want to
/// instrument either "the CPU pool" or "every CUDA pool" rather than a
/// single device ordinal — and routing CUDA overrides per-ordinal would
/// either need a bounded ordinal table or a lookup mutex on the hot
/// path.
///
/// Throws `std::invalid_argument` if `kind` is outside the declared
/// enumerators. `enum class` is not a closed set in C++ — a malformed
/// value can arrive from a memcpy / FFI / out-of-range cast, and we
/// must not index the slot table out-of-bounds in that case.
std::atomic<Allocator*>& override_slot(Device::Kind kind) {
    static std::atomic<Allocator*> slots[kNumDeviceKinds]{};
    const auto idx = static_cast<std::size_t>(kind);
    if (idx >= static_cast<std::size_t>(kNumDeviceKinds)) {
        throw std::invalid_argument("ctorch: unknown Device::Kind");
    }
    return slots[idx];
}

} // namespace

#if defined(CTORCH_HAS_CUDA)
namespace {

/// Number of CUDA devices visible to this process, queried once. Returns 0
/// if there is no driver or no devices — that case makes every CUDA index
/// invalid and the lookup throws below.
int cuda_visible_device_count() {
    static const int kCount = []() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) {
            return 0;
        }
        return n;
    }();
    return kCount;
}

/// Returns the caching allocator that owns CUDA device \p index.
/// Constructs lazily; the slot vector is sized once at first call so a
/// pathologically large `index` cannot grow the table without bound.
detail::CudaCachingAllocator* cuda_allocator_for(int index) {
    if (index < 0) {
        throw std::invalid_argument("ctorch::default_allocator: negative CUDA device index");
    }
    const int device_count = cuda_visible_device_count();
    if (device_count == 0) {
        throw std::runtime_error(
            "ctorch::default_allocator: no CUDA devices visible (driver missing or "
            "cudaGetDeviceCount failed)");
    }
    if (index >= device_count) {
        throw std::out_of_range(
            "ctorch::default_allocator: CUDA device index " + std::to_string(index) +
            " out of range (cudaGetDeviceCount() = " + std::to_string(device_count) + ")");
    }

    static std::mutex mu;
    static std::vector<std::unique_ptr<detail::CudaCachingAllocator>> table(
        static_cast<std::size_t>(device_count));
    std::lock_guard<std::mutex> lock(mu);
    auto& slot = table[static_cast<std::size_t>(index)];
    if (!slot) {
        slot = std::make_unique<detail::CudaCachingAllocator>(index);
    }
    return slot.get();
}

} // namespace
#endif // CTORCH_HAS_CUDA

Allocator* default_allocator(Device device) {
    if (auto* override = override_slot(device.kind).load(std::memory_order_relaxed)) {
        return override;
    }
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

Allocator* set_default_allocator(Device device, Allocator* allocator) {
    return override_slot(device.kind).exchange(allocator, std::memory_order_acq_rel);
}

} // namespace ctorch
