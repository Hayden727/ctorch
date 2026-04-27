//===- src/ops/binary_ops_cpu.cpp - CPU element-wise binary ops -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU kernels for add/sub/mul/div (free + in-place), plus the public
/// front-door free functions, plus dispatch-table registration. One TU
/// for all four ops via functor injection — see Issue 03 §4.3.
///
//===----------------------------------------------------------------------===//

#include "ctorch/ops/elementwise.h"

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"
#include "ctorch/type_promotion.h"

#include "ops/broadcast.h"
#include "ops/cast_cpu.h"
#include "ops/functors.h"
#include "ops/tensor_iter.h"

#include <cstdint>

namespace ctorch {

namespace {

// ---- Templated kernels ----------------------------------------------------

template <class T, class Op>
void binary_kernel_cpu(const Tensor& a, const Tensor& b, Tensor& out, Op op) {
    if (ops::can_use_contiguous_path(a, b, out)) {
        const auto* ap = static_cast<const T*>(a.storage().data()) + a.offset();
        const auto* bp = static_cast<const T*>(b.storage().data()) + b.offset();
        auto* op_out = static_cast<T*>(out.storage().data()) + out.offset();
        const auto n = out.numel();
        #pragma omp simd
        for (std::int64_t i = 0; i < n; ++i) {
            op_out[i] = op(ap[i], bp[i]);
        }
        return;
    }
    const auto br = ops::broadcast_two(a, b);
    const auto ctx = ops::make_binary_indexer(a, b, out, br);
    const auto* a_base = static_cast<const T*>(a.storage().data());
    const auto* b_base = static_cast<const T*>(b.storage().data());
    auto* out_base = static_cast<T*>(out.storage().data());
    ops::for_each_n_binary(ctx, [&](std::int64_t a_off, std::int64_t b_off,
                                    std::int64_t out_off) {
        out_base[out_off] = op(a_base[a_off], b_base[b_off]);
    });
}

template <class Op>
void binary_dispatch_numeric(const Tensor& a, const Tensor& b, Tensor& out, Op op) {
    switch (out.dtype()) {
    case dtype::float32:
        binary_kernel_cpu<float>(a, b, out, op);
        break;
    case dtype::float64:
        binary_kernel_cpu<double>(a, b, out, op);
        break;
    case dtype::int32:
        binary_kernel_cpu<std::int32_t>(a, b, out, op);
        break;
    case dtype::int64:
        binary_kernel_cpu<std::int64_t>(a, b, out, op);
        break;
    case dtype::bool_:
        throw DTypeError("ctorch: binary arithmetic on bool is not supported "
                         "(promote to int32 first)");
    case dtype::bfloat16:
        throw DTypeError("ctorch: bfloat16 not supported in Issue 03");
    }
}

// ---- Backend functions registered with the dispatcher --------------------

void add_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::AddF{});
}
void sub_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::SubF{});
}
void mul_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::MulF{});
}
void div_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::DivF{});
}

void add_inplace_cpu(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::AddF{});
}
void sub_inplace_cpu(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::SubF{});
}
void mul_inplace_cpu(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::MulF{});
}
void div_inplace_cpu(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::DivF{});
}

struct CPUBinaryRegistrar {
    CPUBinaryRegistrar() {
        dispatch::register_op<op::AddOp>(Device::Kind::CPU, &add_cpu);
        dispatch::register_op<op::SubOp>(Device::Kind::CPU, &sub_cpu);
        dispatch::register_op<op::MulOp>(Device::Kind::CPU, &mul_cpu);
        dispatch::register_op<op::DivOp>(Device::Kind::CPU, &div_cpu);
        dispatch::register_op<op::AddInplaceOp>(Device::Kind::CPU, &add_inplace_cpu);
        dispatch::register_op<op::SubInplaceOp>(Device::Kind::CPU, &sub_inplace_cpu);
        dispatch::register_op<op::MulInplaceOp>(Device::Kind::CPU, &mul_inplace_cpu);
        dispatch::register_op<op::DivInplaceOp>(Device::Kind::CPU, &div_inplace_cpu);
    }
};
const CPUBinaryRegistrar kCpuBinaryRegistrar{};

// ---- Front-door helpers ---------------------------------------------------

Tensor maybe_cast(const Tensor& t, dtype target) {
    if (t.dtype() == target) {
        return t;
    }
    if (t.device().is_cpu()) {
        return ops::cast_cpu(t, target);
    }
    // CUDA cast not implemented yet; require explicit cast on CUDA inputs.
    throw DTypeError("ctorch: implicit dtype promotion on CUDA is not yet "
                     "supported — explicitly cast operands");
}

bool same_view(const Tensor& a, const Tensor& b) {
    return a.storage().data() == b.storage().data() && a.offset() == b.offset() &&
           a.shape() == b.shape() && a.stride() == b.stride() && a.dtype() == b.dtype();
}

void check_devices_match(const Tensor& a, const Tensor& b, const char* op) {
    if (a.device() != b.device()) {
        throw DeviceError(std::string("ctorch::") + op +
                          ": operands are on different devices");
    }
}

template <class OpKey>
Tensor binary_front(const Tensor& a, const Tensor& b, const char* name) {
    check_devices_match(a, b, name);
    const auto promoted = promote_types(a.dtype(), b.dtype());
    Tensor a2 = maybe_cast(a, promoted);
    Tensor b2 = maybe_cast(b, promoted);
    const auto br = ops::broadcast_two(a2, b2);
    Tensor out(br.out_shape, promoted, a.device());
    dispatch::call<OpKey>(a.device().kind, a2, b2, out);
    return out;
}

template <class OpKey>
Tensor& binary_inplace_front(Tensor& a, const Tensor& b, const char* name) {
    check_devices_match(a, b, name);
    const auto promoted = promote_types(a.dtype(), b.dtype());
    if (a.dtype() != promoted) {
        throw DTypeError(std::string("ctorch::") + name +
                         ": in-place output dtype must equal the promoted "
                         "dtype of the operands");
    }
    const auto br = ops::broadcast_two(a, b);
    if (br.out_shape != a.shape()) {
        throw ShapeError(std::string("ctorch::") + name +
                         ": in-place output shape must equal the broadcast "
                         "result shape (no implicit reshape)");
    }
    Tensor b2 = maybe_cast(b, promoted);
    if (ops::may_overlap(a, b2) && !same_view(a, b2)) {
        throw AliasError(std::string("ctorch::") + name +
                         ": in-place output aliases a non-trivial view of "
                         "the right-hand operand");
    }
    dispatch::call<OpKey>(a.device().kind, a, b2);
    return a;
}

} // namespace

// ---- Public free functions ------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) { return binary_front<op::AddOp>(a, b, "add"); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary_front<op::SubOp>(a, b, "sub"); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary_front<op::MulOp>(a, b, "mul"); }
Tensor div(const Tensor& a, const Tensor& b) { return binary_front<op::DivOp>(a, b, "div"); }

Tensor& add_(Tensor& a, const Tensor& b) { return binary_inplace_front<op::AddInplaceOp>(a, b, "add_"); }
Tensor& sub_(Tensor& a, const Tensor& b) { return binary_inplace_front<op::SubInplaceOp>(a, b, "sub_"); }
Tensor& mul_(Tensor& a, const Tensor& b) { return binary_inplace_front<op::MulInplaceOp>(a, b, "mul_"); }
Tensor& div_(Tensor& a, const Tensor& b) { return binary_inplace_front<op::DivInplaceOp>(a, b, "div_"); }

} // namespace ctorch
