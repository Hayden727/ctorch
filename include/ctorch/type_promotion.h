//===- include/ctorch/type_promotion.h - dtype promotion rules -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// PyTorch-style type promotion for binary ops. The supported dtypes for
/// Issue 03 are `bool_`, `int32`, `int64`, `float32`, `float64`. `bfloat16`
/// is recognised by the table but always raises `DTypeError` — arithmetic
/// on it is reserved for the mixed-precision milestone.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_TYPE_PROMOTION_H
#define CTORCH_TYPE_PROMOTION_H

#include "ctorch/dtype.h"

namespace ctorch {

/// Returns the dtype that `a op b` should produce for any element-wise op
/// with PyTorch-style promotion. Throws `DTypeError` if either operand is
/// `bfloat16` (out of scope for Issue 03).
///
/// The rule, in order of precedence, is the standard promotion lattice:
///     bool_ < int32 < int64 < float32 < float64
/// with the additional convention that `int* + float32 -> float32`.
dtype promote_types(dtype a, dtype b);

} // namespace ctorch

#endif // CTORCH_TYPE_PROMOTION_H
