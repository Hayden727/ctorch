//===- src/ops/reduction.h - Internal reduction helpers --------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Axis canonicalisation, output-shape helper, and dtype-rule helpers
/// shared between the CPU and CUDA reduction kernels and the front-door
/// free functions in `src/ops/reduction.cpp`.
///
/// `ReductionAxes` is a fixed-size POD so the CUDA kernels can take it
/// by value without dynamic allocation, the same trick `BinaryIndexer`
/// uses in `src/ops/tensor_iter.h`.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_REDUCTION_INTERNAL_H
#define CTORCH_OPS_REDUCTION_INTERNAL_H

#include "ctorch/dtype.h"
#include "ctorch/tensor.h"

#include "ops/tensor_iter.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ctorch::ops {

/// Marks which input dimensions a reduction collapses. `reduce[d]`
/// is true iff dimension `d` is reduced; `rank` is the number of
/// input dimensions. The two pre-computed numels accelerate the
/// kernel's accumulator initialisation and divisor (for `mean`).
struct ReductionAxes {
    std::array<bool, kMaxRank> reduce{};
    int rank = 0;
    std::int64_t reduced_numel = 1;
    std::int64_t kept_numel = 1;
};

/// Canonicalises a list of axes against `x.shape()`. Negative axes are
/// added to ndim; duplicate or out-of-range axes throw `ShapeError`.
/// An empty `dims` vector marks every axis as reduced (whole-tensor
/// reduction). Throws `ShapeError` if `x.shape().size() > kMaxRank`.
ReductionAxes canonicalise(const Tensor& x, std::vector<std::int64_t> dims);

/// Single-axis canonicalisation used by `max(x, dim)` /
/// `argmax(x, dim)`. Returns the non-negative axis index. Throws
/// `ShapeError` if `dim` is out of range.
int canonicalise_single(const Tensor& x, std::int64_t dim);

/// Computes the output shape of a reduction. With `keepdim=true` each
/// reduced dim is replaced by 1; with `keepdim=false` reduced dims are
/// dropped. The 0-d case (every axis reduced, `keepdim=false`) returns
/// an empty shape.
std::vector<std::int64_t> reduced_shape(const Tensor& x, const ReductionAxes& ax, bool keepdim);

/// Output-shape helper for single-axis reductions (max/min/argmax/argmin)
/// that operate on a non-negative axis index.
std::vector<std::int64_t> reduced_shape_single(const Tensor& x, int axis, bool keepdim);

/// dtype rule for `sum` / `prod` (issue 09 §F7): bool / int*
/// promote to int64; floating dtypes pass through. `bfloat16` throws
/// `DTypeError` (matches the existing element-wise policy).
dtype reduce_sum_prod_dtype(dtype in);

/// Throws `DTypeError` if `in` is not a floating dtype. Used by
/// `mean` to require an explicit cast on integer inputs.
void require_float_for_mean(dtype in, const char* name);

/// Throws `DTypeError` if `in` is `bfloat16` (out of scope for issue 09)
/// — used by every reduction's front-door before any kernel is selected.
void reject_bfloat16(dtype in, const char* name);

} // namespace ctorch::ops

#endif // CTORCH_OPS_REDUCTION_INTERNAL_H
