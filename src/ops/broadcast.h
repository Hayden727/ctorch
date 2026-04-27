//===- src/ops/broadcast.h - NumPy-style shape broadcasting ----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Computes the broadcasted output shape of two tensors and the
/// virtually-expanded strides each input must use to read into that shape.
/// Strides are reported in **elements**, not bytes, with size-1 broadcast
/// dimensions assigned a stride of 0 so the same element is read many
/// times — no copy of the input is ever required.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_BROADCAST_H
#define CTORCH_OPS_BROADCAST_H

#include "ctorch/tensor.h"

#include <cstdint>
#include <vector>

namespace ctorch::ops {

struct BroadcastResult {
    std::vector<std::int64_t> out_shape;
    std::vector<std::int64_t> a_stride; ///< padded + zero-stride expanded, in elements
    std::vector<std::int64_t> b_stride;
};

/// Right-aligns the two input shapes, pads the shorter with leading 1s, and
/// for each dimension takes the max of the two sizes. Throws `ShapeError`
/// if any dimension has incompatible sizes (i.e. neither equal nor 1).
BroadcastResult broadcast_two(const Tensor& a, const Tensor& b);

} // namespace ctorch::ops

#endif // CTORCH_OPS_BROADCAST_H
