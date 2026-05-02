//===- include/ctorch/ops/linalg.h - Linear algebra op API ----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Public surface for BLAS-backed linear algebra ops. `transpose` is a
/// zero-copy view (pure metadata mutation, same family as `permute`).
/// `matmul` is dispatched on device through the standard op-keys table
/// and routes to OpenBLAS / Apple Accelerate (CPU) or cuBLAS (CUDA).
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_LINALG_H
#define CTORCH_OPS_LINALG_H

#include "ctorch/tensor.h"

namespace ctorch {

/// Zero-copy transpose: swaps two axes of `x`. Negative `dim0` / `dim1`
/// are normalised against `x.shape().size()`. `dim0 == dim1` returns
/// `x` unchanged. Throws `ShapeError` if either axis is out of range.
Tensor transpose(const Tensor& x, int dim0, int dim1);

/// Matrix multiplication, PyTorch's `torch.matmul` semantics:
/// - 1-D × 1-D ⇒ 0-D scalar (dot product).
/// - 1-D × 2-D ⇒ 1-D row vector × matrix.
/// - 2-D × 1-D ⇒ matrix × column vector ⇒ 1-D.
/// - 2-D × 2-D ⇒ standard GEMM.
/// - N-D × M-D (≥2) ⇒ batched GEMM with broadcasting on leading
///   batch dims.
///
/// Supported dtypes: `float32`, `float64`. Mixed dtypes promote per
/// the standard table (issue #3 §F2). Integer / bool inputs raise
/// `DTypeError`. Cross-device pairs raise `DeviceError`. Inner-dim
/// mismatch raises `ShapeError` with both shapes in the message.
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace ctorch

#endif // CTORCH_OPS_LINALG_H
