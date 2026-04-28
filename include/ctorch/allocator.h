//===- include/ctorch/allocator.h - Allocator interface --------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Stable allocator interface that Storage uses to acquire and release device
/// memory. Concrete implementations (CPU pool, CUDA caching) live in src/.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_ALLOCATOR_H
#define CTORCH_ALLOCATOR_H

#include "ctorch/device.h"

#include <cstddef>

namespace ctorch {

/// Pluggable per-device allocator. Implementations must be safe to call
/// concurrently from multiple threads.
class Allocator {
  public:
    virtual ~Allocator() = default;

    Allocator() = default;
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

    /// Returns a pointer to a buffer of at least \p bytes bytes. The returned
    /// pointer is suitable for any element type ctorch supports (see N1).
    virtual void* allocate(std::size_t bytes) = 0;

    /// Returns a buffer previously obtained from `allocate`. The size must
    /// match the original request.
    virtual void deallocate(void* p, std::size_t bytes) = 0;
};

/// Returns the process-wide default allocator for \p device. The returned
/// pointer is owned by the runtime; callers must not delete it. Lifetime is
/// at least until program termination.
///
/// CUDA allocators are routed by `Device::index` so each ordinal owns a
/// distinct caching pool. This matters under multi-GPU: a tensor tagged
/// `Device::cuda(1)` must be backed by memory that lives on device 1.
///
/// If a caller has installed an override via `set_default_allocator`, that
/// override is returned instead. CUDA overrides are keyed only by
/// `Device::Kind`, not by ordinal — every CUDA device shares one slot.
Allocator* default_allocator(Device device);

/// Installs \p allocator as the override returned by `default_allocator`
/// for \p device's kind, and returns the previous override (or `nullptr`
/// if none). Passing `nullptr` removes any installed override and
/// restores the built-in pool.
///
/// The installed allocator must outlive every Storage that may be backed
/// by it; this hook is intended for tests and instrumentation rather than
/// as a stable user-facing API. The lookup is a single relaxed atomic
/// load so the no-override hot path is unaffected.
Allocator* set_default_allocator(Device device, Allocator* allocator);

} // namespace ctorch

#endif // CTORCH_ALLOCATOR_H
