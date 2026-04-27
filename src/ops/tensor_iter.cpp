//===- src/ops/tensor_iter.cpp - Indexer + alias detection -----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ops/tensor_iter.h"

#include "ctorch/dtype.h"
#include "ctorch/errors.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace ctorch::ops {

namespace {

void copy_with_pad(std::array<std::int64_t, kMaxRank>& dst, const std::vector<std::int64_t>& src) {
    if (src.size() > kMaxRank) {
        throw ShapeError("ctorch: tensor rank exceeds kMaxRank=8");
    }
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i];
    }
}

std::int64_t product(const std::array<std::int64_t, kMaxRank>& a, int rank) {
    std::int64_t p = 1;
    for (int i = 0; i < rank; ++i) {
        p *= a[i];
    }
    return p;
}

} // namespace

BinaryIndexer make_binary_indexer(const Tensor& a, const Tensor& b,
                                  const Tensor& out, const BroadcastResult& br) {
    const std::size_t r = br.out_shape.size();
    if (r > kMaxRank) {
        throw ShapeError("ctorch: broadcast rank exceeds kMaxRank=8");
    }
    BinaryIndexer ctx;
    ctx.rank = static_cast<int>(r);
    copy_with_pad(ctx.shape, br.out_shape);
    copy_with_pad(ctx.a_stride, br.a_stride);
    copy_with_pad(ctx.b_stride, br.b_stride);
    copy_with_pad(ctx.out_stride, out.stride());
    ctx.n = product(ctx.shape, ctx.rank);
    ctx.a_offset_elems = a.offset();
    ctx.b_offset_elems = b.offset();
    ctx.out_offset_elems = out.offset();
    return ctx;
}

UnaryIndexer make_unary_indexer(const Tensor& in, const Tensor& out) {
    const std::size_t r = in.shape().size();
    if (r > kMaxRank) {
        throw ShapeError("ctorch: tensor rank exceeds kMaxRank=8");
    }
    UnaryIndexer ctx;
    ctx.rank = static_cast<int>(r);
    copy_with_pad(ctx.shape, in.shape());
    copy_with_pad(ctx.in_stride, in.stride());
    copy_with_pad(ctx.out_stride, out.stride());
    ctx.n = product(ctx.shape, ctx.rank);
    ctx.in_offset_elems = in.offset();
    ctx.out_offset_elems = out.offset();
    return ctx;
}

bool can_use_contiguous_path(const Tensor& a, const Tensor& b, const Tensor& out) {
    return a.is_contiguous() && b.is_contiguous() && out.is_contiguous() &&
           a.shape() == b.shape() && a.shape() == out.shape() &&
           a.dtype() == out.dtype() && b.dtype() == out.dtype();
}

bool can_use_contiguous_path(const Tensor& in, const Tensor& out) {
    return in.is_contiguous() && out.is_contiguous() && in.shape() == out.shape();
}

bool may_overlap(const Tensor& dst, const Tensor& src) {
    if (!dst.defined() || !src.defined()) {
        return false;
    }
    if (dst.storage().data() != src.storage().data()) {
        return false;
    }
    const auto byte_range = [](const Tensor& t) {
        const auto elem_size = static_cast<std::int64_t>(size_of(t.dtype()));
        const std::int64_t base_bytes = t.offset() * elem_size;
        std::int64_t extent_bytes = 0;
        const auto& shape = t.shape();
        const auto& stride = t.stride();
        for (std::size_t d = 0; d < shape.size(); ++d) {
            if (shape[d] > 1 && stride[d] != 0) {
                extent_bytes += (shape[d] - 1) * std::abs(stride[d]) * elem_size;
            }
        }
        return std::pair<std::int64_t, std::int64_t>{base_bytes, base_bytes + extent_bytes + elem_size};
    };
    const auto [d_lo, d_hi] = byte_range(dst);
    const auto [s_lo, s_hi] = byte_range(src);
    return d_lo < s_hi && s_lo < d_hi;
}

} // namespace ctorch::ops
