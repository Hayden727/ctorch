//===- src/ops/reduction_iter.cpp - Reduction plan construction -----------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ops/reduction_iter.h"

#include "ctorch/tensor.h"

#include "ops/reduction.h"
#include "ops/tensor_iter.h"

#include <cstdint>

namespace ctorch::ops {

ReductionPlan make_reduction_plan(const Tensor& in, const Tensor& out, const ReductionAxes& ax) {
    ReductionPlan p;
    p.in_offset_elems = in.offset();
    p.out_offset_elems = out.offset();
    p.kept_numel = ax.kept_numel;
    p.reduced_numel = ax.reduced_numel;

    const auto& in_shape = in.shape();
    const auto& in_stride = in.stride();
    const auto& out_stride = out.stride();

    int kept_idx = 0;
    int reduced_idx = 0;
    int out_dim = 0;
    for (int d = 0; d < ax.rank; ++d) {
        const auto u = static_cast<std::size_t>(d);
        const std::int64_t dim = in_shape[u];
        const std::int64_t stride = in_stride[u];
        if (ax.reduce[u]) {
            p.shape_reduced[static_cast<std::size_t>(reduced_idx)] = dim;
            p.stride_in_reduced[static_cast<std::size_t>(reduced_idx)] = stride;
            ++reduced_idx;
        } else {
            p.shape_kept[static_cast<std::size_t>(kept_idx)] = dim;
            p.stride_in_kept[static_cast<std::size_t>(kept_idx)] = stride;
            // Output's stride for this kept dim is whatever `out` uses
            // for its own row-major layout. With `keepdim==false` the
            // output skips collapsed dims, so its rank is `rank_kept`
            // and its strides line up 1:1 with kept dims. With
            // `keepdim==true` the output retains singleton dims at
            // collapsed positions (stride=0 effectively, but we never
            // walk those because reduced dims are not in the kept
            // odometer); we have to skip them to find the matching
            // output stride for this kept dim.
            p.stride_out[static_cast<std::size_t>(kept_idx)] =
                out_stride[static_cast<std::size_t>(out_dim)];
            ++kept_idx;
        }
        // Advance the output-dim cursor: with keepdim each input dim
        // maps to one output dim; without keepdim, only kept dims do.
        if (out_stride.size() == in_shape.size()) {
            ++out_dim; // keepdim==true: every input dim has an output slot
        } else if (!ax.reduce[u]) {
            ++out_dim; // keepdim==false: only kept dims have output slots
        }
    }
    p.rank_kept = kept_idx;
    p.rank_reduced = reduced_idx;
    return p;
}

} // namespace ctorch::ops
