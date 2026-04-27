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

/// Returns the process-wide default allocator for \p kind. The returned
/// pointer is owned by the runtime; callers must not delete it. Lifetime is
/// at least until program termination.
Allocator* default_allocator(Device::Kind kind);

} // namespace ctorch

#endif // CTORCH_ALLOCATOR_H
