//===- src/ops/reduction_ops_cpu.cpp - CPU reduction kernels -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU kernels for sum / mean / prod / max / min / argmax / argmin,
/// plus the public front-door free functions and dispatch-table
/// registration. Single TU per Issue #9 §4.2: one templated kernel per
/// reduction family, with the op (`SumF` / `ProdF` / `MaxF` / `MinF`)
/// injected as a functor so CPU and CUDA arithmetic stay byte-identical.
///
//===----------------------------------------------------------------------===//

#include "ctorch/ops/reduction.h"

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "ops/reduction.h"
#include "ops/reduction_functors.h"
#include "ops/reduction_iter.h"
#include "ops/tensor_iter.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ctorch {

namespace {

using ops::ReductionAxes;
using ops::ReductionPlan;

// ---------- Generic reduction kernel --------------------------------------

template <class T, class Acc, class Op, class OutT>
void run_reduction(const Tensor& in, Tensor& out, const ReductionAxes& ax, Op /*tag*/) {
    const auto plan = ops::make_reduction_plan(in, out, ax);
    const auto* in_base = static_cast<const T*>(in.storage().data());
    auto* out_base = static_cast<OutT*>(out.storage().data());

    std::array<std::int64_t, ops::kMaxRank> kidx{};
    std::int64_t in_kept_off = plan.in_offset_elems;
    std::int64_t out_off = plan.out_offset_elems;

    for (std::int64_t k = 0; k < plan.kept_numel; ++k) {
        Acc acc = Op::template identity<Acc>();
        // Inner odometer: walk the reduced subspace.
        std::array<std::int64_t, ops::kMaxRank> ridx{};
        std::int64_t in_red_off = 0;
        for (std::int64_t r = 0; r < plan.reduced_numel; ++r) {
            const T v = in_base[in_kept_off + in_red_off];
            Op::template apply<Acc, T>(acc, v);
            for (int d = plan.rank_reduced - 1; d >= 0; --d) {
                const auto u = static_cast<std::size_t>(d);
                ++ridx[u];
                in_red_off += plan.stride_in_reduced[u];
                if (ridx[u] < plan.shape_reduced[u]) {
                    break;
                }
                ridx[u] = 0;
                in_red_off -= plan.stride_in_reduced[u] * plan.shape_reduced[u];
            }
        }
        out_base[out_off] = static_cast<OutT>(acc);
        // Outer odometer: advance kept dims. The 0-d-output case
        // (rank_kept==0) is handled by the loop bound: kept_numel==1.
        for (int d = plan.rank_kept - 1; d >= 0; --d) {
            const auto u = static_cast<std::size_t>(d);
            ++kidx[u];
            in_kept_off += plan.stride_in_kept[u];
            out_off += plan.stride_out[u];
            if (kidx[u] < plan.shape_kept[u]) {
                break;
            }
            kidx[u] = 0;
            in_kept_off -= plan.stride_in_kept[u] * plan.shape_kept[u];
            out_off -= plan.stride_out[u] * plan.shape_kept[u];
        }
    }
}

// ---------- sum ----------------------------------------------------------

void sum_cpu(const Tensor& in, Tensor& out, const ReductionAxes& ax) {
    // Output dtype is dictated by the front-door (issue 09 §F7):
    //   bool/int* -> int64; float passthrough.
    switch (in.dtype()) {
    case dtype::float32:
        // Wide accumulator to mitigate fp32 drift vs CUDA tree-reduce.
        run_reduction<float, double, ops::SumF, float>(in, out, ax, ops::SumF{});
        break;
    case dtype::float64:
        run_reduction<double, double, ops::SumF, double>(in, out, ax, ops::SumF{});
        break;
    case dtype::int32:
        run_reduction<std::int32_t, std::int64_t, ops::SumF, std::int64_t>(in, out, ax,
                                                                           ops::SumF{});
        break;
    case dtype::int64:
        run_reduction<std::int64_t, std::int64_t, ops::SumF, std::int64_t>(in, out, ax,
                                                                           ops::SumF{});
        break;
    case dtype::bool_:
        // bool storage is `unsigned char` (size_of(bool_)==1).
        run_reduction<unsigned char, std::int64_t, ops::SumF, std::int64_t>(in, out, ax,
                                                                            ops::SumF{});
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::sum: bfloat16 reductions are not supported");
    }
}

// ---------- prod ---------------------------------------------------------

void prod_cpu(const Tensor& in, Tensor& out, const ReductionAxes& ax) {
    switch (in.dtype()) {
    case dtype::float32:
        run_reduction<float, double, ops::ProdF, float>(in, out, ax, ops::ProdF{});
        break;
    case dtype::float64:
        run_reduction<double, double, ops::ProdF, double>(in, out, ax, ops::ProdF{});
        break;
    case dtype::int32:
        run_reduction<std::int32_t, std::int64_t, ops::ProdF, std::int64_t>(in, out, ax,
                                                                            ops::ProdF{});
        break;
    case dtype::int64:
        run_reduction<std::int64_t, std::int64_t, ops::ProdF, std::int64_t>(in, out, ax,
                                                                            ops::ProdF{});
        break;
    case dtype::bool_:
        run_reduction<unsigned char, std::int64_t, ops::ProdF, std::int64_t>(in, out, ax,
                                                                             ops::ProdF{});
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::prod: bfloat16 reductions are not supported");
    }
}

// ---------- mean ---------------------------------------------------------

// `mean = sum / reduced_numel`. The front-door has already verified the
// dtype is floating, so we run a same-dtype sum here and divide in
// place. For the empty-slice case (`reduced_numel == 0`) both branches
// honour PyTorch's "0/0 ⇒ NaN" contract.
template <class T> void mean_kernel(const Tensor& in, Tensor& out, const ReductionAxes& ax) {
    run_reduction<T, double, ops::SumF, T>(in, out, ax, ops::SumF{});
    auto* p = static_cast<T*>(out.storage().data()) + out.offset();
    const std::int64_t n = out.numel();
    if (ax.reduced_numel == 0) {
        const T nan = std::numeric_limits<T>::quiet_NaN();
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = nan;
        }
        return;
    }
    const T inv = static_cast<T>(1) / static_cast<T>(ax.reduced_numel);
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] *= inv;
    }
}

void mean_cpu(const Tensor& in, Tensor& out, const ReductionAxes& ax) {
    switch (in.dtype()) {
    case dtype::float32:
        mean_kernel<float>(in, out, ax);
        break;
    case dtype::float64:
        mean_kernel<double>(in, out, ax);
        break;
    case dtype::int32:
    case dtype::int64:
    case dtype::bool_:
        // The front-door guards integer input; if we ever reach the
        // kernel with one it is a programming error, not user input.
        throw DTypeError("ctorch::mean: integer dtype reached the kernel "
                         "(front-door should have rejected it)");
    case dtype::bfloat16:
        throw DTypeError("ctorch::mean: bfloat16 reductions are not supported");
    }
}

// ---------- registrar ----------------------------------------------------

struct CPUReductionRegistrar {
    CPUReductionRegistrar() {
        dispatch::register_op<op::SumOp>(Device::Kind::CPU, &sum_cpu);
        dispatch::register_op<op::ProdOp>(Device::Kind::CPU, &prod_cpu);
        dispatch::register_op<op::MeanOp>(Device::Kind::CPU, &mean_cpu);
    }
};
const CPUReductionRegistrar kCpuReductionRegistrar{};

// ---------- front-door helpers -------------------------------------------

template <class OpKey>
Tensor sumlike_front(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim,
                     const char* name) {
    ops::reject_bfloat16(x.dtype(), name);
    const auto ax = ops::canonicalise(x, std::move(dims));
    const dtype out_dt = ops::reduce_sum_prod_dtype(x.dtype());
    Tensor out(ops::reduced_shape(x, ax, keepdim), out_dt, x.device());
    dispatch::call<OpKey>(x.device().kind, x, out, ax);
    return out;
}

Tensor mean_front(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim, const char* name) {
    ops::reject_bfloat16(x.dtype(), name);
    ops::require_float_for_mean(x.dtype(), name);
    const auto ax = ops::canonicalise(x, std::move(dims));
    Tensor out(ops::reduced_shape(x, ax, keepdim), x.dtype(), x.device());
    dispatch::call<op::MeanOp>(x.device().kind, x, out, ax);
    return out;
}

} // namespace

// ---------- public free functions ----------------------------------------

Tensor sum(const Tensor& x) { return sumlike_front<op::SumOp>(x, {}, false, "sum"); }
Tensor sum(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim) {
    return sumlike_front<op::SumOp>(x, std::move(dims), keepdim, "sum");
}

Tensor prod(const Tensor& x) { return sumlike_front<op::ProdOp>(x, {}, false, "prod"); }
Tensor prod(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim) {
    return sumlike_front<op::ProdOp>(x, std::move(dims), keepdim, "prod");
}

Tensor mean(const Tensor& x) { return mean_front(x, {}, false, "mean"); }
Tensor mean(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim) {
    return mean_front(x, std::move(dims), keepdim, "mean");
}

} // namespace ctorch
