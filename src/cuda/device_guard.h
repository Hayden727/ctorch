//===- src/cuda/device_guard.h - RAII current-device guard ----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scoped RAII helper that pins the calling thread to a target CUDA device
/// for the lifetime of the guard and restores the prior device on exit.
/// Internal use only — every CUDA runtime call that touches a specific
/// device (cudaMalloc, cudaMemset, cudaMemcpy when the device pointer is
/// device-bound) should be issued under one of these guards so allocator
/// internals never leak into caller state.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_SRC_CUDA_DEVICE_GUARD_H
#define CTORCH_SRC_CUDA_DEVICE_GUARD_H

#include <cuda_runtime.h>
#include <stdexcept>

namespace ctorch::cuda {

class DeviceGuard {
  public:
    explicit DeviceGuard(int target) {
        cudaError_t err = cudaGetDevice(&prev_);
        if (err != cudaSuccess) {
            throw std::runtime_error("ctorch::cuda::DeviceGuard: cudaGetDevice failed");
        }
        if (prev_ != target) {
            err = cudaSetDevice(target);
            if (err != cudaSuccess) {
                throw std::runtime_error("ctorch::cuda::DeviceGuard: cudaSetDevice failed");
            }
            changed_ = true;
        }
    }

    ~DeviceGuard() {
        if (changed_) {
            // Best-effort restore. A throwing destructor would terminate;
            // a bad restore here only affects callers who expect the prior
            // device to still be valid.
            (void)cudaSetDevice(prev_);
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&) = delete;
    DeviceGuard& operator=(DeviceGuard&&) = delete;

  private:
    int prev_ = 0;
    bool changed_ = false;
};

} // namespace ctorch::cuda

#endif // CTORCH_SRC_CUDA_DEVICE_GUARD_H
