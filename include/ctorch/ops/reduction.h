//===- include/ctorch/ops/reduction.h - Reduction op API -------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Free functions for tensor reductions (sum, mean, prod, max, min,
/// argmax, argmin). PyTorch-style: negative axes are normalised, empty
/// `dims` reduces over every axis (whole-tensor reduction), and `keepdim`
/// either drops or replaces reduced dimensions with size 1.
///
/// dtype rules — issue 09 §F7:
///   * `sum`, `prod` of bool / int* promote to int64 (matches PyTorch).
///   * `mean` requires a floating dtype; integer input throws DTypeError.
///   * `max`, `min` preserve the input dtype.
///   * `argmax`, `argmin` always return int64.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_REDUCTION_H
#define CTORCH_OPS_REDUCTION_H

#include "ctorch/errors.h"
#include "ctorch/tensor.h"

#include <cstdint>
#include <vector>

namespace ctorch {

/// Returned by single-axis `max` / `min` — values plus their argmax /
/// argmin indices in the reduced dimension. Matches PyTorch's
/// `torch.max(x, dim=)` shape contract.
struct ValuesIndices {
    Tensor values;
    Tensor indices;
};

// ---------- whole-tensor reductions ----------

/// Reduce every axis. Output is a 0-d tensor (scalar).
Tensor sum(const Tensor& x);
Tensor mean(const Tensor& x);
Tensor prod(const Tensor& x);
Tensor max(const Tensor& x);
Tensor min(const Tensor& x);

// ---------- multi-axis reductions ----------

/// Reduce the axes listed in \p dims. Empty `dims` ⇒ reduce all axes.
/// Negative axes are normalised against `ndim`. Duplicate or
/// out-of-range axes throw `ShapeError`.
Tensor sum(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim = false);
Tensor mean(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim = false);
Tensor prod(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim = false);
Tensor max(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim = false);
Tensor min(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim = false);

// ---------- single-axis max/min returning values + indices ----------

ValuesIndices max(const Tensor& x, std::int64_t dim, bool keepdim = false);
ValuesIndices min(const Tensor& x, std::int64_t dim, bool keepdim = false);

// ---------- index-only reductions ----------

/// Single-axis argmax / argmin returning int64 indices. PyTorch's
/// "first occurrence wins" tie-breaking applies.
Tensor argmax(const Tensor& x, std::int64_t dim, bool keepdim = false);
Tensor argmin(const Tensor& x, std::int64_t dim, bool keepdim = false);

} // namespace ctorch

#endif // CTORCH_OPS_REDUCTION_H
