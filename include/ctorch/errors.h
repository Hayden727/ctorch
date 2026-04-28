//===- include/ctorch/errors.h - Public exception hierarchy ----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Public exception types thrown by ctorch ops. All derive from
/// `ctorch::Error` (itself a `std::runtime_error`) so callers can catch
/// either the specific failure mode or any ctorch error in one block.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_ERRORS_H
#define CTORCH_ERRORS_H

#include <stdexcept>
#include <string>

namespace ctorch {

/// Base class for all ctorch-specific exceptions. Catch this to handle any
/// ctorch failure uniformly.
class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// Thrown when tensor shapes are incompatible (broadcast failure, view
/// mismatch, in-place output shape mismatch, etc.).
class ShapeError : public Error {
  public:
    using Error::Error;
};

/// Thrown when a dtype is unsupported, cannot be promoted, or cannot be
/// cast safely (e.g. promoting `bfloat16` in Issue 03's scope).
class DTypeError : public Error {
  public:
    using Error::Error;
};

/// Thrown when an operation receives operands on incompatible devices, or
/// when a CUDA op is invoked in a build without CUDA support.
class DeviceError : public Error {
  public:
    using Error::Error;
};

/// Thrown by in-place ops when the destination tensor would alias a
/// non-trivial view of an input. Conservative: false positives are
/// preferred to silently writing into the wrong memory.
class AliasError : public Error {
  public:
    using Error::Error;
};

} // namespace ctorch

#endif // CTORCH_ERRORS_H
