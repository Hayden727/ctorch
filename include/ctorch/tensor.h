//===- include/ctorch/tensor.h - Tensor handle -----------------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tensor is a small, copy-cheap handle wrapping shared TensorImpl metadata
/// (shape / stride / offset / dtype) plus a refcounted Storage. Views and
/// permutations produce new Tensors that share the same Storage.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_TENSOR_H
#define CTORCH_TENSOR_H

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/storage.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ctorch {

namespace detail {

struct TensorImpl {
    Storage storage;
    std::int64_t offset = 0; ///< Offset in **elements**, not bytes.
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> stride;
    dtype dt = dtype::float32;
};

} // namespace detail

class Tensor {
  public:
    /// Allocates a fresh, contiguous, zero-initialized tensor of the given
    /// shape and dtype on \p d.
    Tensor(std::vector<std::int64_t> shape, dtype dt, Device d);

    Tensor() = default;

    const std::vector<std::int64_t>& shape() const { return impl_->shape; }
    const std::vector<std::int64_t>& stride() const { return impl_->stride; }
    std::int64_t offset() const { return impl_->offset; }
    ::ctorch::dtype dtype() const { return impl_->dt; }
    Device device() const { return impl_->storage.device(); }

    Storage& storage() { return impl_->storage; }
    const Storage& storage() const { return impl_->storage; }

    /// Total element count. Empty shape (`{}`) is a 0-d scalar with numel 1.
    std::int64_t numel() const;

    /// True iff the stride pattern is the canonical row-major one for the
    /// current shape, i.e. `t.view(t.shape())` would not need to copy.
    bool is_contiguous() const;

    /// Returns a view sharing storage with `*this`. Throws if the strides do
    /// not permit a no-copy reshape to \p new_shape.
    Tensor view(std::vector<std::int64_t> new_shape) const;

    /// PyTorch-style: tries `view`; on incompatibility falls back to
    /// `contiguous().view(new_shape)`.
    Tensor reshape(std::vector<std::int64_t> new_shape) const;

    /// Reorders dimensions by \p dims. Never copies; only stride and shape
    /// are rearranged.
    Tensor permute(std::vector<std::int64_t> dims) const;

    /// Returns `*this` if already contiguous; otherwise materializes a fresh
    /// row-major copy with its own Storage.
    Tensor contiguous() const;

    /// Returns a tensor on \p d. If \p d equals the current device, returns
    /// `*this` (storage is shared). Otherwise allocates fresh storage on \p d
    /// and copies the bytes; round-trip CPU↔CUDA is byte-identical.
    Tensor to(Device d) const;

    bool defined() const { return static_cast<bool>(impl_); }

  private:
    std::shared_ptr<detail::TensorImpl> impl_;

    explicit Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {}
};

} // namespace ctorch

#endif // CTORCH_TENSOR_H
