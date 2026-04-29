//===- src/ops/matmul.cpp - matmul front-door + shape rules ---*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Public free function `ctorch::matmul`. Validates devices / dtypes,
/// promotes the operand dtype, allocates the output, and dispatches to
/// the per-device backend registered against `op::MatmulOp`.
///
//===----------------------------------------------------------------------===//

#include "ctorch/ops/linalg.h"

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"
#include "ctorch/type_promotion.h"

#include "ops/cast_cpu.h"
#include "ops/matmul_shape.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ctorch {

namespace {

struct MatmulRegistrar {
    MatmulRegistrar() {
        dispatch::register_op<op::MatmulOp>(Device::Kind::CPU, &ops::matmul_cpu);
#if defined(CTORCH_HAS_CUDA)
        dispatch::register_op<op::MatmulOp>(Device::Kind::CUDA, &ops::matmul_cuda);
#endif
    }
};
const MatmulRegistrar kMatmulRegistrar{};

void reject_non_float(dtype dt) {
    if (dt != dtype::float32 && dt != dtype::float64) {
        throw DTypeError("ctorch::matmul: requires floating dtype inputs (got " +
                         std::to_string(static_cast<int>(dt)) +
                         "); cast integer / bool inputs to float32 or float64 first");
    }
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw ShapeError("ctorch::matmul: undefined input tensor");
    }
    if (a.device() != b.device()) {
        throw DeviceError("ctorch::matmul: lhs is on " + std::string(a.device().is_cuda() ? "cuda" : "cpu") +
                          ", rhs is on " + std::string(b.device().is_cuda() ? "cuda" : "cpu") +
                          " — both inputs must live on the same device");
    }

    const dtype promoted = promote_types(a.dtype(), b.dtype());
    reject_non_float(promoted);

    // Promote operand dtypes if needed. For CUDA inputs that need
    // promotion we'd need a CUDA cast kernel — that's not in scope for
    // this milestone; reject and ask the user to cast first.
    auto cast_if_needed = [&](const Tensor& t) -> Tensor {
        if (t.dtype() == promoted) {
            return t;
        }
        if (!t.device().is_cpu()) {
            throw DTypeError("ctorch::matmul: implicit dtype promotion on CUDA inputs is not "
                             "supported; cast both operands to the same dtype first");
        }
        return ops::cast_cpu(t, promoted);
    };
    const Tensor a_p = cast_if_needed(a);
    const Tensor b_p = cast_if_needed(b);

    // Materialise to contiguous so the backends can index with a
    // single base pointer + batch offsets. CPU's `.contiguous()` is
    // free when already contiguous; CUDA's still throws when asked to
    // strided-copy on-device, so document the limitation here.
    auto materialise = [](const Tensor& t) -> Tensor {
        if (t.is_contiguous() && t.offset() == 0) {
            return t;
        }
        if (t.device().is_cpu()) {
            return t.contiguous();
        }
        throw DeviceError("ctorch::matmul: non-contiguous CUDA operands are not yet "
                          "supported; call .contiguous() on the operand first");
    };
    const Tensor a_c = materialise(a_p);
    const Tensor b_c = materialise(b_p);

    // Plan validates inner-dim compatibility and returns the
    // user-visible output shape (with 1-D promotions squeezed).
    ops::MatmulPlan plan = ops::plan_matmul(a_c, b_c);

    Tensor out(plan.out_shape, promoted, a.device());

    if (plan.M == 0 || plan.N == 0 || plan.a_offsets.empty()) {
        // Empty result — no GEMM calls needed; output is already
        // zero-initialised by Tensor's ctor.
        return out;
    }

    dispatch::call<op::MatmulOp>(a.device().kind, a_c, b_c, out);
    return out;
}

} // namespace ctorch
