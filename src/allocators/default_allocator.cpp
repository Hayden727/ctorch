//===- src/allocators/default_allocator.cpp - Allocator registry ----------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Process-wide default allocator lookup. Returns a stable pointer per
/// device kind; concrete instances are function-local statics so they live
/// as long as the program does.
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"

#include "allocators/cpu_pool.h"

#include <stdexcept>

#if defined(CTORCH_HAS_CUDA)
#include "allocators/cuda_caching.h"
#endif

namespace ctorch {

Allocator* default_allocator(Device::Kind kind) {
    switch (kind) {
    case Device::Kind::CPU: {
        static detail::CpuPoolAllocator cpu;
        return &cpu;
    }
    case Device::Kind::CUDA: {
#if defined(CTORCH_HAS_CUDA)
        static detail::CudaCachingAllocator cuda;
        return &cuda;
#else
        throw std::runtime_error(
            "ctorch::default_allocator: CUDA backend not built (CTORCH_CUDA=OFF)");
#endif
    }
    }
    throw std::invalid_argument("ctorch::default_allocator: unknown Device::Kind");
}

} // namespace ctorch
