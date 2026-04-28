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

// ---------- max / min (values only, multi-axis or whole-tensor) ----------

template <class Op>
void maxmin_cpu_dispatch(const Tensor& in, Tensor& out, const ReductionAxes& ax, const char* name) {
    switch (in.dtype()) {
    case dtype::float32:
        run_reduction<float, float, Op, float>(in, out, ax, Op{});
        break;
    case dtype::float64:
        run_reduction<double, double, Op, double>(in, out, ax, Op{});
        break;
    case dtype::int32:
        run_reduction<std::int32_t, std::int32_t, Op, std::int32_t>(in, out, ax, Op{});
        break;
    case dtype::int64:
        run_reduction<std::int64_t, std::int64_t, Op, std::int64_t>(in, out, ax, Op{});
        break;
    case dtype::bool_:
        run_reduction<unsigned char, unsigned char, Op, unsigned char>(in, out, ax, Op{});
        break;
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 reductions are not "
                         "supported");
    }
}

void max_val_cpu(const Tensor& in, Tensor& out, const ReductionAxes& ax) {
    maxmin_cpu_dispatch<ops::MaxF>(in, out, ax, "max");
}
void min_val_cpu(const Tensor& in, Tensor& out, const ReductionAxes& ax) {
    maxmin_cpu_dispatch<ops::MinF>(in, out, ax, "min");
}

// ---------- max / min with indices (single axis) -----------------------

// Single-axis kernel that records both the running best value and its
// position along the reduced axis. `vals_out` may be null for the
// argmax/argmin paths that only need the index.
template <class T, class Op>
void run_axis_with_idx_cpu(const Tensor& in, Tensor* vals_out, Tensor& idx_out, int axis) {
    const auto& shape = in.shape();
    const auto& stride = in.stride();
    const int rank = static_cast<int>(shape.size());
    const std::int64_t reduced_size = shape[static_cast<std::size_t>(axis)];
    const std::int64_t reduced_stride = stride[static_cast<std::size_t>(axis)];

    // Build a ReductionAxes with the single axis flagged so we can reuse
    // the shared plan helper for the kept-axis odometer.
    ReductionAxes ax{};
    ax.rank = rank;
    ax.reduce[static_cast<std::size_t>(axis)] = true;
    ax.reduced_numel = reduced_size;
    ax.kept_numel = 1;
    for (int d = 0; d < rank; ++d) {
        if (d != axis) {
            ax.kept_numel *= shape[static_cast<std::size_t>(d)];
        }
    }
    const Tensor& reference_out = vals_out != nullptr ? *vals_out : idx_out;
    const auto plan = ops::make_reduction_plan(in, reference_out, ax);

    const auto* in_base = static_cast<const T*>(in.storage().data());
    T* vals_base = vals_out != nullptr ? static_cast<T*>(vals_out->storage().data()) : nullptr;
    auto* idx_base = static_cast<std::int64_t*>(idx_out.storage().data());

    std::array<std::int64_t, ops::kMaxRank> kidx{};
    std::int64_t in_kept_off = plan.in_offset_elems;
    std::int64_t out_off = plan.out_offset_elems;
    for (std::int64_t k = 0; k < plan.kept_numel; ++k) {
        // First element seeds the running best so first-occurrence-wins
        // semantics apply uniformly even with a NaN-only slice.
        T best = in_base[in_kept_off];
        std::int64_t best_idx = 0;
        for (std::int64_t r = 1; r < reduced_size; ++r) {
            const T v = in_base[in_kept_off + r * reduced_stride];
            if (Op::template should_replace<T>(best, v)) {
                best = v;
                best_idx = r;
            }
        }
        if (vals_base != nullptr) {
            vals_base[out_off] = best;
        }
        idx_base[out_off] = best_idx;

        // Advance kept odometer (same loop as run_reduction).
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

template <class Op>
void maxmin_with_idx_cpu_dispatch(const Tensor& in, Tensor& vals, Tensor& idx, int axis,
                                  const char* name) {
    switch (in.dtype()) {
    case dtype::float32:
        run_axis_with_idx_cpu<float, Op>(in, &vals, idx, axis);
        break;
    case dtype::float64:
        run_axis_with_idx_cpu<double, Op>(in, &vals, idx, axis);
        break;
    case dtype::int32:
        run_axis_with_idx_cpu<std::int32_t, Op>(in, &vals, idx, axis);
        break;
    case dtype::int64:
        run_axis_with_idx_cpu<std::int64_t, Op>(in, &vals, idx, axis);
        break;
    case dtype::bool_:
        run_axis_with_idx_cpu<unsigned char, Op>(in, &vals, idx, axis);
        break;
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 reductions are not "
                         "supported");
    }
}

void max_val_idx_cpu(const Tensor& in, Tensor& vals, Tensor& idx, int axis) {
    maxmin_with_idx_cpu_dispatch<ops::MaxF>(in, vals, idx, axis, "max");
}
void min_val_idx_cpu(const Tensor& in, Tensor& vals, Tensor& idx, int axis) {
    maxmin_with_idx_cpu_dispatch<ops::MinF>(in, vals, idx, axis, "min");
}

template <class Op>
void argmaxmin_cpu_dispatch(const Tensor& in, Tensor& idx, int axis, const char* name) {
    switch (in.dtype()) {
    case dtype::float32:
        run_axis_with_idx_cpu<float, Op>(in, nullptr, idx, axis);
        break;
    case dtype::float64:
        run_axis_with_idx_cpu<double, Op>(in, nullptr, idx, axis);
        break;
    case dtype::int32:
        run_axis_with_idx_cpu<std::int32_t, Op>(in, nullptr, idx, axis);
        break;
    case dtype::int64:
        run_axis_with_idx_cpu<std::int64_t, Op>(in, nullptr, idx, axis);
        break;
    case dtype::bool_:
        run_axis_with_idx_cpu<unsigned char, Op>(in, nullptr, idx, axis);
        break;
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 reductions are not "
                         "supported");
    }
}

void argmax_cpu(const Tensor& in, Tensor& idx, int axis) {
    argmaxmin_cpu_dispatch<ops::MaxF>(in, idx, axis, "argmax");
}
void argmin_cpu(const Tensor& in, Tensor& idx, int axis) {
    argmaxmin_cpu_dispatch<ops::MinF>(in, idx, axis, "argmin");
}

} // namespace

#if defined(CTORCH_HAS_CUDA)
// Implemented in src/ops/reduction_ops_cuda.cu. Referencing this
// symbol from here forces the linker to pull the .cu TU into the
// final binary, which in turn brings the CUDA-side dispatch
// registrations along (same trick as binary_ops_cpu.cpp).
extern "C" void ctorch_register_cuda_reduction_ops();
#endif

namespace {

struct CPUReductionRegistrar {
    CPUReductionRegistrar() {
        dispatch::register_op<op::SumOp>(Device::Kind::CPU, &sum_cpu);
        dispatch::register_op<op::ProdOp>(Device::Kind::CPU, &prod_cpu);
        dispatch::register_op<op::MeanOp>(Device::Kind::CPU, &mean_cpu);
        dispatch::register_op<op::MaxValOp>(Device::Kind::CPU, &max_val_cpu);
        dispatch::register_op<op::MinValOp>(Device::Kind::CPU, &min_val_cpu);
        dispatch::register_op<op::MaxValIdxOp>(Device::Kind::CPU, &max_val_idx_cpu);
        dispatch::register_op<op::MinValIdxOp>(Device::Kind::CPU, &min_val_idx_cpu);
        dispatch::register_op<op::ArgmaxOp>(Device::Kind::CPU, &argmax_cpu);
        dispatch::register_op<op::ArgminOp>(Device::Kind::CPU, &argmin_cpu);
#if defined(CTORCH_HAS_CUDA)
        ctorch_register_cuda_reduction_ops();
#endif
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

template <class OpKey>
Tensor maxmin_value_front(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim,
                          const char* name) {
    ops::reject_bfloat16(x.dtype(), name);
    const auto ax = ops::canonicalise(x, std::move(dims));
    if (ax.reduced_numel == 0) {
        throw ShapeError(std::string("ctorch::") + name +
                         ": cannot reduce a zero-element slice (operation has no identity)");
    }
    Tensor out(ops::reduced_shape(x, ax, keepdim), x.dtype(), x.device());
    dispatch::call<OpKey>(x.device().kind, x, out, ax);
    return out;
}

template <class OpKey>
ValuesIndices maxmin_with_idx_front(const Tensor& x, std::int64_t dim, bool keepdim,
                                    const char* name) {
    ops::reject_bfloat16(x.dtype(), name);
    const int axis = ops::canonicalise_single(x, dim);
    if (x.shape()[static_cast<std::size_t>(axis)] == 0) {
        throw ShapeError(std::string("ctorch::") + name +
                         ": cannot reduce a zero-length axis (operation has no identity)");
    }
    const auto out_shape = ops::reduced_shape_single(x, axis, keepdim);
    Tensor vals(out_shape, x.dtype(), x.device());
    Tensor idx(out_shape, dtype::int64, x.device());
    dispatch::call<OpKey>(x.device().kind, x, vals, idx, axis);
    return ValuesIndices{std::move(vals), std::move(idx)};
}

template <class OpKey>
Tensor argmaxmin_front(const Tensor& x, std::int64_t dim, bool keepdim, const char* name) {
    ops::reject_bfloat16(x.dtype(), name);
    const int axis = ops::canonicalise_single(x, dim);
    if (x.shape()[static_cast<std::size_t>(axis)] == 0) {
        throw ShapeError(std::string("ctorch::") + name +
                         ": cannot reduce a zero-length axis (operation has no identity)");
    }
    Tensor idx(ops::reduced_shape_single(x, axis, keepdim), dtype::int64, x.device());
    dispatch::call<OpKey>(x.device().kind, x, idx, axis);
    return idx;
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

Tensor max(const Tensor& x) { return maxmin_value_front<op::MaxValOp>(x, {}, false, "max"); }
Tensor max(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim) {
    return maxmin_value_front<op::MaxValOp>(x, std::move(dims), keepdim, "max");
}
ValuesIndices max(const Tensor& x, std::int64_t dim, bool keepdim) {
    return maxmin_with_idx_front<op::MaxValIdxOp>(x, dim, keepdim, "max");
}

Tensor min(const Tensor& x) { return maxmin_value_front<op::MinValOp>(x, {}, false, "min"); }
Tensor min(const Tensor& x, std::vector<std::int64_t> dims, bool keepdim) {
    return maxmin_value_front<op::MinValOp>(x, std::move(dims), keepdim, "min");
}
ValuesIndices min(const Tensor& x, std::int64_t dim, bool keepdim) {
    return maxmin_with_idx_front<op::MinValIdxOp>(x, dim, keepdim, "min");
}

Tensor argmax(const Tensor& x, std::int64_t dim, bool keepdim) {
    return argmaxmin_front<op::ArgmaxOp>(x, dim, keepdim, "argmax");
}
Tensor argmin(const Tensor& x, std::int64_t dim, bool keepdim) {
    return argmaxmin_front<op::ArgminOp>(x, dim, keepdim, "argmin");
}

} // namespace ctorch
