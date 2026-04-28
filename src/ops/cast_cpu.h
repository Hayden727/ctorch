//===- src/ops/cast_cpu.h - In-library dtype cast helper -------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal CPU dtype-cast utility used by the binary-op front-door when
/// PyTorch-style promotion requires widening one of the operands. Not a
/// public op — see Issue 03 §4.2 for context.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_CAST_CPU_H
#define CTORCH_OPS_CAST_CPU_H

#include "ctorch/dtype.h"
#include "ctorch/tensor.h"

namespace ctorch::ops {

/// Returns `t` if it already has dtype `target`; otherwise allocates a
/// fresh contiguous tensor of dtype `target` and copies elements with a
/// static_cast. The result is always contiguous and on the same device
/// as `t`. Implemented for CPU tensors only.
Tensor cast_cpu(const Tensor& t, dtype target);

} // namespace ctorch::ops

#endif // CTORCH_OPS_CAST_CPU_H
