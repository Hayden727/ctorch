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

#if defined(CTORCH_HAS_CUDA)
#include "cuda/device_guard.h"
#endif

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

void reject_non_float(dtype dt, const char* operand) {
    if (dt != dtype::float32 && dt != dtype::float64) {
        throw DTypeError(std::string("ctorch::matmul: requires floating dtype inputs (") + operand +
                         " has dtype " + std::to_string(static_cast<int>(dt)) +
                         "); cast integer / bool inputs to float32 or float64 first");
    }
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined()) {
        throw ShapeError("ctorch::matmul: undefined input tensor");
    }
    if (a.device() != b.device()) {
        throw DeviceError("ctorch::matmul: lhs is on " +
                          std::string(a.device().is_cuda() ? "cuda" : "cpu") + ", rhs is on " +
                          std::string(b.device().is_cuda() ? "cuda" : "cpu") +
                          " — both inputs must live on the same device");
    }

    // Reject integer / bool operands BEFORE promotion. Otherwise
    // `int32 × float32` would silently widen to `float32` and pass.
    // The documented contract is that any non-float operand throws.
    reject_non_float(a.dtype(), "lhs");
    reject_non_float(b.dtype(), "rhs");
    const dtype promoted = promote_types(a.dtype(), b.dtype());

    // Helper: pin the calling thread to the operand's CUDA device for
    // the duration of any sub-call (`Tensor::to`, `cast_cpu`, etc.) so
    // multi-GPU callers don't accidentally allocate / launch on the
    // wrong context. No-op on CPU operands.
    auto with_device = [](const Tensor& t, auto&& body) -> Tensor {
#if defined(CTORCH_HAS_CUDA)
        if (t.device().is_cuda()) {
            cuda::DeviceGuard guard(t.device().index);
            return body();
        }
#endif
        return body();
    };

    // Promote operand dtypes if needed. CPU casts go through `cast_cpu`
    // directly; CUDA casts round-trip through CPU because we don't yet
    // have a CUDA cast kernel. Slow, but correct and consistent with
    // the documented promotion rules — both devices accept the same
    // input pairs.
    auto cast_if_needed = [&](const Tensor& t) -> Tensor {
        if (t.dtype() == promoted) {
            return t;
        }
        if (t.device().is_cpu()) {
            return ops::cast_cpu(t, promoted);
        }
        const Device dev = t.device();
        return with_device(t, [&] { return ops::cast_cpu(t.to(Device::cpu()), promoted).to(dev); });
    };
    const Tensor a_p = cast_if_needed(a);
    const Tensor b_p = cast_if_needed(b);

    // Materialise to contiguous so the backends can index with a
    // single base pointer + batch offsets. CPU's `.contiguous()` is
    // free when already contiguous; for non-contiguous CUDA operands
    // we round-trip through CPU since on-device strided materialisation
    // is not yet implemented (acceptable as a correctness fallback —
    // perf optimisation deferred to the bench milestone).
    auto materialise = [&](const Tensor& t) -> Tensor {
        if (t.is_contiguous() && t.offset() == 0) {
            return t;
        }
        if (t.device().is_cpu()) {
            return t.contiguous();
        }
        const Device dev = t.device();
        return with_device(t, [&] { return t.to(Device::cpu()).contiguous().to(dev); });
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
