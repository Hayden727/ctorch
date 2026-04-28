//===- src/ops/tensor_iter.h - Strided indexer + alias check ---*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Fixed-size strided iterator context shared between CPU and CUDA op
/// implementations, plus an alias-overlap detector used by in-place ops.
///
/// The iterator is intentionally pass-by-value-into-CUDA-kernel friendly:
/// the rank is bounded by `kMaxRank` and all per-dimension data lives in
/// `std::array`s rather than `std::vector`s.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_TENSOR_ITER_H
#define CTORCH_OPS_TENSOR_ITER_H

#include "ctorch/tensor.h"
#include "ops/broadcast.h"

#include <array>
#include <cstdint>

namespace ctorch::ops {

/// Maximum number of dimensions handled by the fixed-size iterator. The
/// upstream Tensor API accepts arbitrary rank, so this bound has to cover
/// any reasonable user input — 16 is generous for ML workloads (PyTorch's
/// TensorIterator caps at 16 too) while keeping each IndexerCtx well under
/// the 4 KB CUDA kernel-parameter limit (4 strides × 16 × 8 bytes ≈ 520 B).
constexpr int kMaxRank = 16;

/// Strided iteration context for an element-wise binary op. Strides are in
/// **elements** (not bytes); `out_stride` is the contiguous row-major
/// stride of the output tensor. `n` is the total number of output elements.
struct BinaryIndexer {
    int rank = 0;
    std::int64_t n = 0;
    std::array<std::int64_t, kMaxRank> shape{};
    std::array<std::int64_t, kMaxRank> a_stride{};
    std::array<std::int64_t, kMaxRank> b_stride{};
    std::array<std::int64_t, kMaxRank> out_stride{};
    std::int64_t a_offset_elems = 0;
    std::int64_t b_offset_elems = 0;
    std::int64_t out_offset_elems = 0;
};

/// Same idea as BinaryIndexer but for unary ops.
struct UnaryIndexer {
    int rank = 0;
    std::int64_t n = 0;
    std::array<std::int64_t, kMaxRank> shape{};
    std::array<std::int64_t, kMaxRank> in_stride{};
    std::array<std::int64_t, kMaxRank> out_stride{};
    std::int64_t in_offset_elems = 0;
    std::int64_t out_offset_elems = 0;
};

/// Build a binary indexer from precomputed broadcast information and the
/// preallocated output tensor.
BinaryIndexer make_binary_indexer(const Tensor& a, const Tensor& b, const Tensor& out,
                                  const BroadcastResult& br);

/// Build a unary indexer for `out = f(in)`.
UnaryIndexer make_unary_indexer(const Tensor& in, const Tensor& out);

/// Returns true iff the binary inputs and output are all contiguous, share
/// the same shape (no broadcast expansion), and have the same dtype as
/// `out`. Hot-loop fast path.
bool can_use_contiguous_path(const Tensor& a, const Tensor& b, const Tensor& out);
bool can_use_contiguous_path(const Tensor& in, const Tensor& out);

/// Returns true iff `dst` and `src` share the same underlying buffer and
/// their byte ranges overlap. Conservative — false positives are
/// preferred to silently writing into the wrong memory.
bool may_overlap(const Tensor& dst, const Tensor& src);

/// CPU odometer for binary ops: invokes `f(a_off, b_off, out_off)` with
/// element offsets relative to each tensor's base for every output index.
template <class F> void for_each_n_binary(const BinaryIndexer& ctx, F&& f) {
    if (ctx.rank == 0) {
        f(ctx.a_offset_elems, ctx.b_offset_elems, ctx.out_offset_elems);
        return;
    }
    std::array<std::int64_t, kMaxRank> idx{};
    std::int64_t a_off = ctx.a_offset_elems;
    std::int64_t b_off = ctx.b_offset_elems;
    std::int64_t out_off = ctx.out_offset_elems;
    for (std::int64_t i = 0; i < ctx.n; ++i) {
        f(a_off, b_off, out_off);
        for (int d = ctx.rank - 1; d >= 0; --d) {
            ++idx[d];
            a_off += ctx.a_stride[d];
            b_off += ctx.b_stride[d];
            out_off += ctx.out_stride[d];
            if (idx[d] < ctx.shape[d]) {
                break;
            }
            idx[d] = 0;
            a_off -= ctx.a_stride[d] * ctx.shape[d];
            b_off -= ctx.b_stride[d] * ctx.shape[d];
            out_off -= ctx.out_stride[d] * ctx.shape[d];
        }
    }
}

/// CPU odometer for unary ops: invokes `f(in_off, out_off)`.
template <class F> void for_each_n_unary(const UnaryIndexer& ctx, F&& f) {
    if (ctx.rank == 0) {
        f(ctx.in_offset_elems, ctx.out_offset_elems);
        return;
    }
    std::array<std::int64_t, kMaxRank> idx{};
    std::int64_t in_off = ctx.in_offset_elems;
    std::int64_t out_off = ctx.out_offset_elems;
    for (std::int64_t i = 0; i < ctx.n; ++i) {
        f(in_off, out_off);
        for (int d = ctx.rank - 1; d >= 0; --d) {
            ++idx[d];
            in_off += ctx.in_stride[d];
            out_off += ctx.out_stride[d];
            if (idx[d] < ctx.shape[d]) {
                break;
            }
            idx[d] = 0;
            in_off -= ctx.in_stride[d] * ctx.shape[d];
            out_off -= ctx.out_stride[d] * ctx.shape[d];
        }
    }
}

} // namespace ctorch::ops

#endif // CTORCH_OPS_TENSOR_ITER_H
