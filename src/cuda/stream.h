//===- src/cuda/stream.h - CUDA stream RAII wrapper ------------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thin RAII wrapper around `cudaStream_t`. Future kernels and the caching
/// allocator key on the stream the work was issued on.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_SRC_CUDA_STREAM_H
#define CTORCH_SRC_CUDA_STREAM_H

#include <cuda_runtime.h>

namespace ctorch::cuda {

/// Owning wrapper around `cudaStream_t`. The default-constructed instance is
/// the implicit default stream (`cudaStreamLegacy`); explicit-constructed
/// instances own a stream they create with `cudaStreamCreate`.
class Stream {
  public:
    Stream() : raw_(cudaStreamLegacy), owned_(false) {}
    static Stream create();

    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;

    cudaStream_t raw() const noexcept { return raw_; }

    /// Block the calling thread until all work issued on this stream
    /// completes.
    void synchronize() const;

  private:
    Stream(cudaStream_t raw, bool owned) : raw_(raw), owned_(owned) {}
    cudaStream_t raw_;
    bool owned_;
};

} // namespace ctorch::cuda

#endif // CTORCH_SRC_CUDA_STREAM_H
