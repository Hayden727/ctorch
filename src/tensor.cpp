//===- src/tensor.cpp - Tensor implementation -----------------------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// View / reshape / permute / contiguous / to-device for the Tensor type.
/// All view operations alias the source Storage; copies happen only when the
/// requested shape cannot be expressed by stride manipulation alone, or when
/// the destination device differs from the source.
///
//===----------------------------------------------------------------------===//

#include "ctorch/tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(CTORCH_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace ctorch {

namespace {

using ShapeT = std::vector<std::int64_t>;

ShapeT contiguous_stride(const ShapeT& shape) {
    ShapeT s(shape.size(), 1);
    if (shape.empty()) {
        return s;
    }
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 2; i >= 0; --i) {
        s[static_cast<std::size_t>(i)] =
            s[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
    }
    return s;
}

std::int64_t shape_numel(const ShapeT& shape) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        if (d < 0) {
            throw std::invalid_argument("ctorch::Tensor: negative dimension");
        }
        n *= d;
    }
    return n;
}

bool is_contiguous_for(const ShapeT& shape, const ShapeT& stride) {
    if (shape.size() != stride.size()) {
        return false;
    }
    if (shape.empty()) {
        return true;
    }
    // A tensor with a zero-sized dimension is trivially contiguous; element
    // ordering is unobservable.
    for (std::int64_t d : shape) {
        if (d == 0) {
            return true;
        }
    }
    std::int64_t expected = 1;
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 1; i >= 0; --i) {
        if (stride[static_cast<std::size_t>(i)] != expected) {
            return false;
        }
        expected *= shape[static_cast<std::size_t>(i)];
    }
    return true;
}

void copy_bytes(void* dst, Device dst_dev, const void* src, Device src_dev, std::size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    if (src_dev.is_cpu() && dst_dev.is_cpu()) {
        std::memcpy(dst, src, nbytes);
        return;
    }
#if defined(CTORCH_HAS_CUDA)
    cudaMemcpyKind kind = cudaMemcpyHostToHost;
    if (src_dev.is_cpu() && dst_dev.is_cuda()) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_dev.is_cuda() && dst_dev.is_cpu()) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_dev.is_cuda() && dst_dev.is_cuda()) {
        kind = cudaMemcpyDeviceToDevice;
    }
    cudaError_t err = cudaMemcpy(dst, src, nbytes, kind);
    if (err != cudaSuccess) {
        throw std::runtime_error("ctorch::Tensor::to: cudaMemcpy failed");
    }
#else
    throw std::runtime_error("ctorch::Tensor::to: CUDA copy requested but build has no CUDA");
#endif
}

// Walk source via shape/stride and write into a contiguous destination buffer.
// Implemented as an iterative odometer over `shape` so it works for arbitrary
// rank without recursion. Both buffers must live on the same device, which
// for now restricts non-contiguous->contiguous materialization to CPU.
void copy_strided_to_contiguous(const std::byte* src, std::byte* dst, const ShapeT& shape,
                                const ShapeT& stride, std::size_t elem_size) {
    if (shape.empty()) {
        std::memcpy(dst, src, elem_size);
        return;
    }
    const std::size_t rank = shape.size();
    ShapeT idx(rank, 0);
    std::int64_t total = shape_numel(shape);

    for (std::int64_t linear = 0; linear < total; ++linear) {
        std::int64_t src_off_elems = 0;
        for (std::size_t d = 0; d < rank; ++d) {
            src_off_elems += idx[d] * stride[d];
        }
        std::memcpy(dst + static_cast<std::size_t>(linear) * elem_size,
                    src + static_cast<std::size_t>(src_off_elems) * elem_size, elem_size);

        // Increment the rightmost index, carrying over.
        for (std::int64_t d = static_cast<std::int64_t>(rank) - 1; d >= 0; --d) {
            auto u = static_cast<std::size_t>(d);
            if (++idx[u] < shape[u]) {
                break;
            }
            idx[u] = 0;
        }
    }
}

} // namespace

Tensor::Tensor(std::vector<std::int64_t> shape, ::ctorch::dtype dt, Device d) {
    auto impl = std::make_shared<detail::TensorImpl>();
    impl->shape = std::move(shape);
    impl->stride = contiguous_stride(impl->shape);
    impl->dt = dt;
    impl->offset = 0;

    std::int64_t n = shape_numel(impl->shape);
    std::size_t nbytes = static_cast<std::size_t>(n) * size_of(dt);
    impl->storage = Storage(nbytes, d);

    impl_ = std::move(impl);
}

std::int64_t Tensor::numel() const { return shape_numel(impl_->shape); }

bool Tensor::is_contiguous() const { return is_contiguous_for(impl_->shape, impl_->stride); }

Tensor Tensor::view(std::vector<std::int64_t> new_shape) const {
    if (!is_contiguous()) {
        // A more permissive rule (PyTorch's strict view rule) would allow
        // certain non-contiguous strides; for now we require contiguity and
        // direct callers needing a copy through reshape().
        throw std::runtime_error(
            "ctorch::Tensor::view: source is not contiguous; use reshape() instead");
    }
    if (shape_numel(new_shape) != numel()) {
        throw std::runtime_error("ctorch::Tensor::view: numel mismatch");
    }
    auto out = std::make_shared<detail::TensorImpl>();
    out->storage = impl_->storage;
    out->offset = impl_->offset;
    out->dt = impl_->dt;
    out->shape = std::move(new_shape);
    out->stride = contiguous_stride(out->shape);
    return Tensor(std::move(out));
}

Tensor Tensor::reshape(std::vector<std::int64_t> new_shape) const {
    if (is_contiguous()) {
        return view(std::move(new_shape));
    }
    return contiguous().view(std::move(new_shape));
}

Tensor Tensor::permute(std::vector<std::int64_t> dims) const {
    const std::size_t rank = impl_->shape.size();
    if (dims.size() != rank) {
        throw std::runtime_error("ctorch::Tensor::permute: dims size mismatch");
    }
    std::vector<bool> seen(rank, false);
    for (std::int64_t d : dims) {
        if (d < 0 || static_cast<std::size_t>(d) >= rank) {
            throw std::runtime_error("ctorch::Tensor::permute: dim out of range");
        }
        if (seen[static_cast<std::size_t>(d)]) {
            throw std::runtime_error("ctorch::Tensor::permute: duplicate dim");
        }
        seen[static_cast<std::size_t>(d)] = true;
    }

    auto out = std::make_shared<detail::TensorImpl>();
    out->storage = impl_->storage;
    out->offset = impl_->offset;
    out->dt = impl_->dt;
    out->shape.resize(rank);
    out->stride.resize(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        auto src = static_cast<std::size_t>(dims[i]);
        out->shape[i] = impl_->shape[src];
        out->stride[i] = impl_->stride[src];
    }
    return Tensor(std::move(out));
}

Tensor Tensor::contiguous() const {
    if (is_contiguous() && impl_->offset == 0) {
        return *this;
    }
    if (!device().is_cpu()) {
        throw std::runtime_error(
            "ctorch::Tensor::contiguous: non-CPU strided materialization not yet supported");
    }
    Tensor out(impl_->shape, impl_->dt, device());
    const std::size_t elem = size_of(impl_->dt);
    const auto* src_base = static_cast<const std::byte*>(impl_->storage.data()) +
                           static_cast<std::size_t>(impl_->offset) * elem;
    auto* dst_base = static_cast<std::byte*>(out.impl_->storage.data());
    copy_strided_to_contiguous(src_base, dst_base, impl_->shape, impl_->stride, elem);
    return out;
}

Tensor Tensor::to(Device d) const {
    if (d == device()) {
        return *this;
    }
    Tensor src = is_contiguous() && impl_->offset == 0 ? *this : contiguous();
    Tensor out(src.impl_->shape, src.impl_->dt, d);
    copy_bytes(out.impl_->storage.data(), d, src.impl_->storage.data(), src.device(),
               src.impl_->storage.nbytes());
    return out;
}

} // namespace ctorch
