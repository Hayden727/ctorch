//===- include/ctorch/ops/elementwise.h - Element-wise op API --*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Free functions for element-wise binary and unary ops, plus operator
/// overloads on Tensor. Inputs are broadcast NumPy/PyTorch-style; the
/// output dtype is the promotion of the input dtypes; the output device
/// matches the (single) input device — cross-device ops throw DeviceError.
///
/// This header is included only by consumers who want arithmetic on
/// Tensor; the core Tensor type itself remains free of operator coupling.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_ELEMENTWISE_H
#define CTORCH_OPS_ELEMENTWISE_H

#include "ctorch/tensor.h"

namespace ctorch {

// ---------- binary ----------

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// ---------- in-place ----------

Tensor& add_(Tensor& a, const Tensor& b);
Tensor& sub_(Tensor& a, const Tensor& b);
Tensor& mul_(Tensor& a, const Tensor& b);
Tensor& div_(Tensor& a, const Tensor& b);

// ---------- unary ----------

Tensor neg(const Tensor& x);
Tensor abs(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);

// ---------- operator overloads ----------

inline Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
inline Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
inline Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
inline Tensor operator-(const Tensor& x) { return neg(x); }

} // namespace ctorch

#endif // CTORCH_OPS_ELEMENTWISE_H
