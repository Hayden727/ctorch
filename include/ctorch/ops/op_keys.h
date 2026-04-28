//===- include/ctorch/ops/op_keys.h - Operator dispatch tags ---*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// One empty tag struct per operator. Each defines `using fn_t = ...;`
/// matching the signature that `dispatch::call<OpKey>` will invoke. Backend
/// implementations register against these tags via
/// `dispatch::register_op<OpKey>(kind, &impl)`.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_OP_KEYS_H
#define CTORCH_OPS_OP_KEYS_H

#include "ctorch/tensor.h"

#include <cstdint>

namespace ctorch::ops {
struct ReductionAxes; // defined in src/ops/reduction.h
} // namespace ctorch::ops

namespace ctorch::op {

// Element-wise binary ops. Output tensor is preallocated by the front-door
// function (it knows the broadcast shape, promoted dtype, and device); the
// kernel just fills it.
using BinaryFn = void (*)(const Tensor&, const Tensor&, Tensor&);

struct AddOp {
    using fn_t = BinaryFn;
};
struct SubOp {
    using fn_t = BinaryFn;
};
struct MulOp {
    using fn_t = BinaryFn;
};
struct DivOp {
    using fn_t = BinaryFn;
};

// In-place variants. The first argument is the destination (also the LHS
// operand). The kernel must not allocate.
using BinaryInplaceFn = void (*)(Tensor&, const Tensor&);

struct AddInplaceOp {
    using fn_t = BinaryInplaceFn;
};
struct SubInplaceOp {
    using fn_t = BinaryInplaceFn;
};
struct MulInplaceOp {
    using fn_t = BinaryInplaceFn;
};
struct DivInplaceOp {
    using fn_t = BinaryInplaceFn;
};

// Element-wise unary ops. Output tensor is preallocated by the front-door
// function with the same shape and dtype as the input.
using UnaryFn = void (*)(const Tensor&, Tensor&);

struct NegOp {
    using fn_t = UnaryFn;
};
struct AbsOp {
    using fn_t = UnaryFn;
};
struct ExpOp {
    using fn_t = UnaryFn;
};
struct LogOp {
    using fn_t = UnaryFn;
};
struct SqrtOp {
    using fn_t = UnaryFn;
};
struct ReluOp {
    using fn_t = UnaryFn;
};
struct SigmoidOp {
    using fn_t = UnaryFn;
};
struct TanhOp {
    using fn_t = UnaryFn;
};

// ---------- reductions ----------

// Multi-axis / whole-tensor reductions. The output tensor is preallocated
// with the post-reduction shape and the dtype dictated by the front-door
// (issue 09 §F7); the kernel writes into it.
using ReduceFn = void (*)(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax);

// Single-axis values+indices reductions (PyTorch's max/min along a dim).
// Both outputs are preallocated by the front-door — `vals` shares dtype
// with the input, `idx` is always int64.
using ReduceWithIdxFn = void (*)(const Tensor& in, Tensor& vals, Tensor& idx, int axis);

// Argmax/argmin: just the int64 index, no values.
using ArgFn = void (*)(const Tensor& in, Tensor& idx, int axis);

struct SumOp {
    using fn_t = ReduceFn;
};
struct MeanOp {
    using fn_t = ReduceFn;
};
struct ProdOp {
    using fn_t = ReduceFn;
};
struct MaxValOp {
    using fn_t = ReduceFn;
};
struct MinValOp {
    using fn_t = ReduceFn;
};
struct MaxValIdxOp {
    using fn_t = ReduceWithIdxFn;
};
struct MinValIdxOp {
    using fn_t = ReduceWithIdxFn;
};
struct ArgmaxOp {
    using fn_t = ArgFn;
};
struct ArgminOp {
    using fn_t = ArgFn;
};

} // namespace ctorch::op

#endif // CTORCH_OPS_OP_KEYS_H
