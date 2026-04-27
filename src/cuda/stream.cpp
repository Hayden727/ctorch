//===- src/cuda/stream.cpp - CUDA stream wrapper impl ---------------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the CUDA stream RAII wrapper.
///
//===----------------------------------------------------------------------===//

#include "cuda/stream.h"

#include <stdexcept>

namespace ctorch::cuda {

Stream Stream::create() {
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
        throw std::runtime_error("ctorch::cuda::Stream::create: cudaStreamCreate failed");
    }
    return Stream(s, /*owned=*/true);
}

Stream::~Stream() {
    if (owned_ && raw_ != nullptr) {
        // Best-effort cleanup; swallowing the error in a destructor is the
        // standard idiom because a throwing destructor would terminate.
        (void)cudaStreamDestroy(raw_);
    }
}

Stream::Stream(Stream&& other) noexcept : raw_(other.raw_), owned_(other.owned_) {
    other.raw_ = cudaStreamLegacy;
    other.owned_ = false;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (owned_ && raw_ != nullptr) {
            (void)cudaStreamDestroy(raw_);
        }
        raw_ = other.raw_;
        owned_ = other.owned_;
        other.raw_ = cudaStreamLegacy;
        other.owned_ = false;
    }
    return *this;
}

void Stream::synchronize() const {
    cudaError_t err = cudaStreamSynchronize(raw_);
    if (err != cudaSuccess) {
        throw std::runtime_error("ctorch::cuda::Stream::synchronize: cudaStreamSynchronize failed");
    }
}

} // namespace ctorch::cuda
