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

    const std::vector<std::int64_t>& shape() const {
        if (!impl_) {
            throw_undefined("shape");
        }
        return impl_->shape;
    }
    const std::vector<std::int64_t>& stride() const {
        if (!impl_) {
            throw_undefined("stride");
        }
        return impl_->stride;
    }
    std::int64_t offset() const {
        if (!impl_) {
            throw_undefined("offset");
        }
        return impl_->offset;
    }
    ::ctorch::dtype dtype() const {
        if (!impl_) {
            throw_undefined("dtype");
        }
        return impl_->dt;
    }
    Device device() const {
        if (!impl_) {
            throw_undefined("device");
        }
        return impl_->storage.device();
    }

    Storage& storage() {
        if (!impl_) {
            throw_undefined("storage");
        }
        return impl_->storage;
    }
    const Storage& storage() const {
        if (!impl_) {
            throw_undefined("storage");
        }
        return impl_->storage;
    }

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

    /// 2-D shorthand for `transpose(*this, 0, 1)`. Throws `ShapeError` when
    /// the tensor is not exactly 2-D — use the free `transpose(x, i, j)`
    /// for arbitrary rank.
    Tensor T() const;

    /// Returns `*this` if already contiguous; otherwise materializes a fresh
    /// row-major copy with its own Storage.
    Tensor contiguous() const;

    /// Returns a tensor on \p d. If \p d equals the current device, returns
    /// `*this` (storage is shared). Otherwise allocates fresh storage on \p d
    /// and copies the bytes; round-trip CPU↔CUDA is byte-identical.
    Tensor to(Device d) const;

    /// Zero-copy slice along \p dim. Negative \p dim, \p start, \p end are
    /// normalised against `shape[dim]`; \p start / \p end are then clamped to
    /// `[0, shape[dim]]` (PyTorch-compatible). \p step must be > 0; otherwise
    /// throws `ShapeError`. The result shares storage with `*this`.
    Tensor slice(int dim, std::int64_t start, std::int64_t end, std::int64_t step = 1) const;

    /// Zero-copy single-element selection along \p dim. The selected dim is
    /// removed from the result (rank goes down by 1). Negative \p index is
    /// normalised against `shape[dim]`; out-of-range throws `ShapeError`.
    Tensor select(int dim, std::int64_t index) const;

    /// Zero-copy narrow: sugar for `slice(dim, start, start + length, 1)`.
    /// Negative \p start is normalised against `shape[dim]`; \p length must be
    /// non-negative and `start + length <= shape[dim]`; otherwise throws
    /// `ShapeError`.
    Tensor narrow(int dim, std::int64_t start, std::int64_t length) const;

    bool defined() const { return static_cast<bool>(impl_); }

  private:
    std::shared_ptr<detail::TensorImpl> impl_;

    explicit Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {}

    /// Out-of-line cold path: keeps the inline accessors small and avoids
    /// pulling <stdexcept> + <string> into every translation unit that
    /// transitively includes this header.
    [[noreturn]] static void throw_undefined(const char* fn);
};

} // namespace ctorch

#endif // CTORCH_TENSOR_H
