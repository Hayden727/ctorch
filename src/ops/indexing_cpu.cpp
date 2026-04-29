//===- src/ops/indexing_cpu.cpp - index_select front-door + CPU kernel ----===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Front-door for `ctorch::index_select` plus the CPU gather kernel and
/// dispatch-table registration. The kernel walks the contiguous output
/// linearly, re-encoding each linear index back into output coordinates;
/// the `dim`-th coordinate is then replaced by `indices[coord]` to load
/// from `src`. Out-of-range indices raise `ShapeError`.
///
//===----------------------------------------------------------------------===//

#include "ctorch/ops/indexing.h"

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ctorch {

namespace {

template <class T, class I>
void run_index_select_cpu(const Tensor& src, int dim, const Tensor& indices, Tensor& out) {
    const auto& src_shape = src.shape();
    const auto& src_stride = src.stride();
    const int rank = static_cast<int>(src_shape.size());

    const std::int64_t src_dim_size = src_shape[static_cast<std::size_t>(dim)];
    const std::int64_t src_dim_stride = src_stride[static_cast<std::size_t>(dim)];

    const std::int64_t n_indices = indices.numel();
    const std::int64_t idx_stride = indices.shape().empty() ? 0 : indices.stride()[0];
    const auto* idx_base = static_cast<const I*>(indices.storage().data()) + indices.offset();

    const auto* src_base = static_cast<const T*>(src.storage().data()) + src.offset();
    auto* out_base = static_cast<T*>(out.storage().data()) + out.offset();

    // Pre-resolve indices into source-axis offsets so the bounds-check and
    // negative normalisation happen exactly once per index entry, not once
    // per output element. Validation runs even when the output is empty
    // (other dims have size zero) so out-of-range indices are still
    // surfaced rather than silently ignored.
    std::vector<std::int64_t> resolved(static_cast<std::size_t>(n_indices));
    for (std::int64_t i = 0; i < n_indices; ++i) {
        std::int64_t v = static_cast<std::int64_t>(idx_base[i * idx_stride]);
        if (v < 0) {
            v += src_dim_size;
        }
        if (v < 0 || v >= src_dim_size) {
            throw ShapeError("ctorch::index_select: index " +
                             std::to_string(static_cast<std::int64_t>(idx_base[i * idx_stride])) +
                             " out of range for dim " + std::to_string(dim) + " of size " +
                             std::to_string(src_dim_size));
        }
        resolved[static_cast<std::size_t>(i)] = v * src_dim_stride;
    }

    const std::int64_t total = out.numel();
    if (total == 0) {
        return;
    }

    // Walk the contiguous output as a coord odometer; for each output
    // element compute the strided source offset (the dim-th coord goes
    // through `resolved`, every other dim follows `src_stride`).
    const auto& out_shape = out.shape();
    std::vector<std::int64_t> coords(static_cast<std::size_t>(rank), 0);
    for (std::int64_t lin = 0; lin < total; ++lin) {
        std::int64_t src_off = 0;
        for (int i = 0; i < rank; ++i) {
            const std::int64_t c = coords[static_cast<std::size_t>(i)];
            if (i == dim) {
                src_off += resolved[static_cast<std::size_t>(c)];
            } else {
                src_off += c * src_stride[static_cast<std::size_t>(i)];
            }
        }
        out_base[lin] = src_base[src_off];

        for (int i = rank - 1; i >= 0; --i) {
            const auto u = static_cast<std::size_t>(i);
            if (++coords[u] < out_shape[u]) {
                break;
            }
            coords[u] = 0;
        }
    }
}

template <class T>
void index_select_dispatch_idx(const Tensor& src, int dim, const Tensor& indices, Tensor& out) {
    switch (indices.dtype()) {
    case dtype::int32:
        run_index_select_cpu<T, std::int32_t>(src, dim, indices, out);
        break;
    case dtype::int64:
        run_index_select_cpu<T, std::int64_t>(src, dim, indices, out);
        break;
    default:
        // Front-door already rejected non-int dtypes; reaching here is a
        // programming error, not user input.
        throw DTypeError("ctorch::index_select: indices dtype must be int32 or int64");
    }
}

void index_select_cpu(const Tensor& src, int dim, const Tensor& indices, Tensor& out) {
    switch (src.dtype()) {
    case dtype::float32:
        index_select_dispatch_idx<float>(src, dim, indices, out);
        break;
    case dtype::float64:
        index_select_dispatch_idx<double>(src, dim, indices, out);
        break;
    case dtype::int32:
        index_select_dispatch_idx<std::int32_t>(src, dim, indices, out);
        break;
    case dtype::int64:
        index_select_dispatch_idx<std::int64_t>(src, dim, indices, out);
        break;
    case dtype::bool_:
        index_select_dispatch_idx<unsigned char>(src, dim, indices, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::index_select: bfloat16 is not supported");
    }
}

} // namespace

#if defined(CTORCH_HAS_CUDA)
extern "C" void ctorch_register_cuda_indexing_ops();
#endif

namespace {

struct CPUIndexingRegistrar {
    CPUIndexingRegistrar() {
        dispatch::register_op<op::IndexSelectOp>(Device::Kind::CPU, &index_select_cpu);
#if defined(CTORCH_HAS_CUDA)
        ctorch_register_cuda_indexing_ops();
#endif
    }
};
const CPUIndexingRegistrar kCpuIndexingRegistrar{};

} // namespace

// ---------- public front-door --------------------------------------------

Tensor index_select(const Tensor& src, int dim, const Tensor& indices) {
    if (!src.defined()) {
        throw ShapeError("ctorch::index_select: src is undefined");
    }
    if (!indices.defined()) {
        throw ShapeError("ctorch::index_select: indices is undefined");
    }
    if (indices.shape().size() != 1) {
        throw ShapeError("ctorch::index_select: indices must be 1-D (got rank " +
                         std::to_string(indices.shape().size()) + ")");
    }
    if (indices.dtype() != dtype::int32 && indices.dtype() != dtype::int64) {
        throw DTypeError("ctorch::index_select: indices dtype must be int32 or int64");
    }
    if (src.device() != indices.device()) {
        throw DeviceError("ctorch::index_select: src and indices must live on the same device");
    }
    if (src.dtype() == dtype::bfloat16) {
        throw DTypeError("ctorch::index_select: bfloat16 is not supported");
    }
    const int rank = static_cast<int>(src.shape().size());
    if (rank == 0) {
        throw ShapeError("ctorch::index_select: cannot index a 0-d tensor");
    }
    const int adj = dim < 0 ? dim + rank : dim;
    if (adj < 0 || adj >= rank) {
        throw ShapeError("ctorch::index_select: dim " + std::to_string(dim) +
                         " out of range for tensor of rank " + std::to_string(rank));
    }
    std::vector<std::int64_t> out_shape = src.shape();
    out_shape[static_cast<std::size_t>(adj)] = indices.numel();
    Tensor out(std::move(out_shape), src.dtype(), src.device());
    // Always dispatch even when `out.numel() == 0`: the kernel still has
    // to validate the index buffer (out-of-range entries are a contract
    // violation regardless of whether they would be dereferenced).
    dispatch::call<op::IndexSelectOp>(src.device().kind, src, adj, indices, out);
    return out;
}

} // namespace ctorch
