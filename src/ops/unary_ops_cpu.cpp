//===- src/ops/unary_ops_cpu.cpp - CPU element-wise unary ops --*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU kernels for neg/abs/exp/log/sqrt/relu/sigmoid/tanh, plus public
/// front-door free functions and dispatch-table registration. One TU
/// for all eight ops via functor injection — see Issue 03 §4.3.
///
//===----------------------------------------------------------------------===//

#include "ctorch/ops/elementwise.h"

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "ops/functors.h"
#include "ops/tensor_iter.h"

#include <cstdint>
#include <string>

namespace ctorch {

namespace {

// ---- Templated kernels ---------------------------------------------------

template <class T, class Op>
void unary_kernel_cpu(const Tensor& in, Tensor& out, Op op) {
    if (ops::can_use_contiguous_path(in, out)) {
        const auto* ip = static_cast<const T*>(in.storage().data()) + in.offset();
        auto* op_out = static_cast<T*>(out.storage().data()) + out.offset();
        const auto n = out.numel();
        #pragma omp simd
        for (std::int64_t i = 0; i < n; ++i) {
            op_out[i] = op(ip[i]);
        }
        return;
    }
    const auto ctx = ops::make_unary_indexer(in, out);
    const auto* in_base = static_cast<const T*>(in.storage().data());
    auto* out_base = static_cast<T*>(out.storage().data());
    ops::for_each_n_unary(ctx, [&](std::int64_t in_off, std::int64_t out_off) {
        out_base[out_off] = op(in_base[in_off]);
    });
}

// Dispatch on dtype for ops that accept any signed numeric type.
template <class Op>
void unary_dispatch_signed_numeric(const Tensor& in, Tensor& out, Op op,
                                   const char* name) {
    switch (out.dtype()) {
    case dtype::float32:
        unary_kernel_cpu<float>(in, out, op);
        break;
    case dtype::float64:
        unary_kernel_cpu<double>(in, out, op);
        break;
    case dtype::int32:
        unary_kernel_cpu<std::int32_t>(in, out, op);
        break;
    case dtype::int64:
        unary_kernel_cpu<std::int64_t>(in, out, op);
        break;
    case dtype::bool_:
        throw DTypeError(std::string("ctorch::") + name +
                         ": not defined on bool — promote to int32 first");
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 not supported in Issue 03");
    }
}

// Dispatch on dtype for transcendental ops (float-only).
template <class Op>
void unary_dispatch_float(const Tensor& in, Tensor& out, Op op, const char* name) {
    switch (out.dtype()) {
    case dtype::float32:
        unary_kernel_cpu<float>(in, out, op);
        break;
    case dtype::float64:
        unary_kernel_cpu<double>(in, out, op);
        break;
    case dtype::int32:
    case dtype::int64:
    case dtype::bool_:
        throw DTypeError(std::string("ctorch::") + name +
                         ": only float32/float64 inputs are supported");
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 not supported in Issue 03");
    }
}

// ---- Backend functions registered with the dispatcher --------------------

void neg_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_signed_numeric(in, out, ops::NegF{}, "neg");
}
void abs_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_signed_numeric(in, out, ops::AbsF{}, "abs");
}
void relu_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_signed_numeric(in, out, ops::ReluF{}, "relu");
}
void exp_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::ExpF{}, "exp");
}
void log_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::LogF{}, "log");
}
void sqrt_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::SqrtF{}, "sqrt");
}
void sigmoid_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::SigmoidF{}, "sigmoid");
}
void tanh_cpu(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::TanhF{}, "tanh");
}

} // namespace

#if defined(CTORCH_HAS_CUDA)
extern "C" void ctorch_register_cuda_unary_ops();
#endif

namespace {

struct CPUUnaryRegistrar {
    CPUUnaryRegistrar() {
        dispatch::register_op<op::NegOp>(Device::Kind::CPU, &neg_cpu);
        dispatch::register_op<op::AbsOp>(Device::Kind::CPU, &abs_cpu);
        dispatch::register_op<op::ReluOp>(Device::Kind::CPU, &relu_cpu);
        dispatch::register_op<op::ExpOp>(Device::Kind::CPU, &exp_cpu);
        dispatch::register_op<op::LogOp>(Device::Kind::CPU, &log_cpu);
        dispatch::register_op<op::SqrtOp>(Device::Kind::CPU, &sqrt_cpu);
        dispatch::register_op<op::SigmoidOp>(Device::Kind::CPU, &sigmoid_cpu);
        dispatch::register_op<op::TanhOp>(Device::Kind::CPU, &tanh_cpu);
#if defined(CTORCH_HAS_CUDA)
        ctorch_register_cuda_unary_ops();
#endif
    }
};
const CPUUnaryRegistrar kCpuUnaryRegistrar{};

// ---- Front-door helpers ---------------------------------------------------

template <class OpKey>
Tensor unary_front(const Tensor& in) {
    Tensor out(in.shape(), in.dtype(), in.device());
    dispatch::call<OpKey>(in.device().kind, in, out);
    return out;
}

} // namespace

// ---- Public free functions ------------------------------------------------

Tensor neg(const Tensor& x) { return unary_front<op::NegOp>(x); }
Tensor abs(const Tensor& x) { return unary_front<op::AbsOp>(x); }
Tensor relu(const Tensor& x) { return unary_front<op::ReluOp>(x); }
Tensor exp(const Tensor& x) { return unary_front<op::ExpOp>(x); }
Tensor log(const Tensor& x) { return unary_front<op::LogOp>(x); }
Tensor sqrt(const Tensor& x) { return unary_front<op::SqrtOp>(x); }
Tensor sigmoid(const Tensor& x) { return unary_front<op::SigmoidOp>(x); }
Tensor tanh(const Tensor& x) { return unary_front<op::TanhOp>(x); }

} // namespace ctorch
