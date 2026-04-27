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
        zero_fill(data_, nbytes_, device_);
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
        allocator = default_allocator(d.kind);
    }
    impl_ = intrusive_ptr<detail::StorageImpl>::make(nbytes, d, allocator);
}

} // namespace ctorch
