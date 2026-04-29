//===- src/ops/reduction_iter.h - Reduction iteration plan -----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Splits a tensor's shape / stride into "kept" and "reduced" halves so
/// the kernel walks the kept-axis subspace in the outer loop and the
/// reduced-axis subspace in the inner loop. Both halves are bounded by
/// `kMaxRank`, so a `ReductionPlan` is a kernel-friendly POD with no
/// heap allocation.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_REDUCTION_ITER_H
#define CTORCH_OPS_REDUCTION_ITER_H

#include "ctorch/tensor.h"

#include "ops/reduction.h"
#include "ops/tensor_iter.h"

#include <array>
#include <cstdint>

namespace ctorch::ops {

/// Two-axis-set odometer: the **kept** axes drive the outer loop (one
/// output element each), the **reduced** axes drive the inner loop
/// (folded into the accumulator). Strides are in elements, lifted from
/// the input tensor; the output is always contiguous so its strides are
/// the canonical row-major ones over the kept-axis shape.
struct ReductionPlan {
    int rank_kept = 0;
    int rank_reduced = 0;
    std::array<std::int64_t, kMaxRank> shape_kept{};
    std::array<std::int64_t, kMaxRank> shape_reduced{};
    std::array<std::int64_t, kMaxRank> stride_in_kept{};
    std::array<std::int64_t, kMaxRank> stride_in_reduced{};
    std::array<std::int64_t, kMaxRank> stride_out{};
    std::int64_t in_offset_elems = 0;
    std::int64_t out_offset_elems = 0;
    std::int64_t kept_numel = 1;
    std::int64_t reduced_numel = 1;
};

/// Build a `ReductionPlan` for `out = reduce(in)` given the
/// canonicalised `ax`. Caller is responsible for ensuring `out` has the
/// kept-axis shape (with or without keepdim==true singletons collapsed)
/// and is contiguous on its own storage.
ReductionPlan make_reduction_plan(const Tensor& in, const Tensor& out, const ReductionAxes& ax);

} // namespace ctorch::ops

#endif // CTORCH_OPS_REDUCTION_ITER_H
