//===- src/storage.cpp - Storage implementation ---------------------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage / StorageImpl implementation. Allocation is delegated to the
/// supplied Allocator; the buffer is zero-initialized at construction time
/// (Issue 02 §AC1).
///
//===----------------------------------------------------------------------===//

#include "ctorch/storage.h"

#include <cstring>
#include <stdexcept>

#if defined(CTORCH_HAS_CUDA)
#include "cuda/device_guard.h"

#include <cuda_runtime.h>
#endif

namespace ctorch {

namespace detail {

namespace {

void zero_fill(void* data, std::size_t nbytes, Device device) {
    if (data == nullptr || nbytes == 0) {
        return;
    }
    if (device.is_cpu()) {
        std::memset(data, 0, nbytes);
        return;
    }
#if defined(CTORCH_HAS_CUDA)
    // The caching allocator restores the caller's current device after
    // cudaMalloc, so by the time we get here the runtime is no longer
    // pinned to `device.index`. cudaMemset uses the current device's
    // context to interpret the pointer; without this guard a non-current
    // CUDA storage would zero-fill against the wrong context and either
    // fail or quietly skip the init.
    cuda::DeviceGuard guard(device.index);
    cudaError_t err = cudaMemset(data, 0, nbytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("ctorch::Storage: cudaMemset failed");
    }
#else
    throw std::runtime_error("ctorch::Storage: CUDA device requested but build has no CUDA");
#endif
}

} // namespace

StorageImpl::StorageImpl(std::size_t nbytes, Device device, Allocator* allocator)
    : nbytes_(nbytes), device_(device), allocator_(allocator) {
    if (allocator_ == nullptr) {
        throw std::invalid_argument("ctorch::Storage: allocator must not be null");
    }
    if (nbytes_ > 0) {
        data_ = allocator_->allocate(nbytes_);
        // If zero_fill throws (e.g. cudaMemset failure), we are still mid-
        // construction, so ~StorageImpl will not run and the buffer would
        // otherwise leak. Hand it back to the allocator and rethrow.
        try {
            zero_fill(data_, nbytes_, device_);
        } catch (...) {
            allocator_->deallocate(data_, nbytes_);
            data_ = nullptr;
            throw;
        }
    }
}

StorageImpl::~StorageImpl() {
    if (data_ != nullptr && allocator_ != nullptr) {
        allocator_->deallocate(data_, nbytes_);
    }
}

} // namespace detail

Storage::Storage(std::size_t nbytes, Device d, Allocator* allocator) {
    if (allocator == nullptr) {
        // Pass the full Device so CUDA allocators see the index, not just
        // the kind. Otherwise a Device::cuda(1) tensor would be served by
        // the same pool as Device::cuda(0).
        allocator = default_allocator(d);
    }
    impl_ = intrusive_ptr<detail::StorageImpl>::make(nbytes, d, allocator);
}

} // namespace ctorch
