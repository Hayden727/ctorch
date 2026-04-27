//===- src/ops/broadcast.cpp - Broadcast shape and strides -----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ops/broadcast.h"

#include "ctorch/errors.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace ctorch::ops {

namespace {

std::string shape_to_string(const std::vector<std::int64_t>& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

} // namespace

BroadcastResult broadcast_two(const Tensor& a, const Tensor& b) {
    const auto& sa = a.shape();
    const auto& sb = b.shape();
    const auto& strA = a.stride();
    const auto& strB = b.stride();

    const std::size_t ra = sa.size();
    const std::size_t rb = sb.size();
    const std::size_t r = std::max(ra, rb);

    BroadcastResult out;
    out.out_shape.resize(r);
    out.a_stride.assign(r, 0);
    out.b_stride.assign(r, 0);

    for (std::size_t i = 0; i < r; ++i) {
        // Right-align: dim i counts from the right. PyTorch convention:
        // shape `[3, 1]` aligned with `[1, 4]` matches dim-by-dim from the
        // tail.
        const std::size_t ia = (i + ra >= r) ? (i + ra - r) : 0;
        const std::size_t ib = (i + rb >= r) ? (i + rb - r) : 0;
        const bool has_a = (i + ra >= r);
        const bool has_b = (i + rb >= r);

        const std::int64_t da = has_a ? sa[ia] : 1;
        const std::int64_t db = has_b ? sb[ib] : 1;

        std::int64_t out_dim;
        if (da == db) {
            out_dim = da;
        } else if (da == 1) {
            out_dim = db;
        } else if (db == 1) {
            out_dim = da;
        } else {
            throw ShapeError("ctorch: cannot broadcast shapes " +
                             shape_to_string(sa) + " and " + shape_to_string(sb));
        }
        out.out_shape[i] = out_dim;

        // Stride for a in dim i: 0 if size 1 (broadcast), else original
        // stride. Same for b. If a doesn't have this dim at all, its stride
        // contribution is 0 (treat as virtual leading 1).
        out.a_stride[i] = (has_a && da == out_dim) ? strA[ia] : 0;
        out.b_stride[i] = (has_b && db == out_dim) ? strB[ib] : 0;
    }
    return out;
}

} // namespace ctorch::ops
