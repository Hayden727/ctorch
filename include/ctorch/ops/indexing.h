//===- include/ctorch/ops/indexing.h - Indexing op API ---------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Free functions that produce **fresh contiguous** tensors selected from
/// `src` along a single axis. Zero-copy view ops (`slice` / `select` /
/// `narrow`) live on `Tensor` itself and share storage; `index_select`
/// here always allocates a new output and dispatches on the source
/// device.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_INDEXING_H
#define CTORCH_OPS_INDEXING_H

#include "ctorch/tensor.h"

#include <cstdint>

namespace ctorch {

/// Gather rows of `src` along `dim` according to `indices`. `indices`
/// must be a 1-D tensor of dtype `int32` or `int64` on the same device
/// as `src`. The output has shape `src.shape()` with `src.shape[dim]`
/// replaced by `indices.numel()`; it is freshly allocated and
/// contiguous. Negative `dim` is normalised against `src.shape().size()`.
/// Out-of-range index values raise `ShapeError`.
Tensor index_select(const Tensor& src, int dim, const Tensor& indices);

} // namespace ctorch

#endif // CTORCH_OPS_INDEXING_H
