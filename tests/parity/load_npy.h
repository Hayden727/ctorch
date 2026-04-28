//===- tests/parity/load_npy.h - Minimal NPY v1.0 reader -------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loads a NumPy `.npy` v1.0 fixture into a fresh contiguous CPU Tensor.
/// Only the subset of dtypes ctorch currently supports is recognised:
///     <f4 → float32, <f8 → float64,
///     <i4 → int32,   <i8 → int64,
///     |b1 → bool_.
/// Other dtypes raise DTypeError with the offending NPY descr in the
/// message — when fp16/bf16/etc. land in a future ctorch milestone, only
/// the translation table grows.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_TESTS_PARITY_LOAD_NPY_H
#define CTORCH_TESTS_PARITY_LOAD_NPY_H

#include "ctorch/tensor.h"

#include <string>

namespace ctorch::parity {

/// Reads `path` and returns its contents as a fresh contiguous CPU Tensor.
/// Throws ctorch::Error or its subclasses on any parse / IO error.
Tensor load_npy(const std::string& path);

} // namespace ctorch::parity

#endif // CTORCH_TESTS_PARITY_LOAD_NPY_H
