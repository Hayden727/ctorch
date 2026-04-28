//===- src/ops/reduction_ops_cuda.cu - CUDA reduction kernels --*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA kernels for reductions. Two kernel families:
///
///   1. **Whole-tensor** (`kept_numel == 1`): two-pass tree reduction.
///      Pass 1 reads the input via the strided odometer, folds into
///      shared memory, and writes one partial per block. Pass 2
///      reduces the partials in a single block.
///
///   2. **Axis** (`kept_numel > 1`): one thread per output element.
///      Each thread decodes its kept-axis offset, then walks the
///      reduced subspace serially with a per-thread odometer kept in
///      registers. Same kernel handles innermost and non-innermost
///      reduced axes; transpose-then-reduce optimisation for the
///      latter is deferred to a follow-up issue (issue #9 §7).
///
///   3. **Axis with indices** (`max(x, dim)` / `min(x, dim)` /
///      `argmax` / `argmin`): like family 2 but tracks
///      `(best_value, best_idx)` along a single reduced axis. The
///      `RecordValue` template parameter lets argmax/argmin reuse the
///      same kernel and skip the value write.
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/ops/reduction.h"
#include "ctorch/tensor.h"

#include "cuda/device_guard.h"
#include "ops/reduction.h"
#include "ops/reduction_functors.h"
#include "ops/reduction_iter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace ctorch {

namespace {

constexpr int kBlockSize = 256;

void check_cuda(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw DeviceError(std::string("ctorch::") + what +
                          ": CUDA error: " + cudaGetErrorString(err));
    }
}

// Decode the linear thread id into the strided input offset for the
// reduced subspace. Splits like the ReductionPlan but only the
// reduced-axis half (the kept half is unused on the whole-tensor path
// where kept_numel==1).
template <class T>
__device__ T load_at_linear(const T* in_base, const ops::ReductionPlan& plan, std::int64_t i) {
    std::int64_t off = plan.in_offset_elems;
    std::int64_t remainder = i;
    for (int d = plan.rank_reduced - 1; d >= 0; --d) {
        const std::int64_t dim = plan.shape_reduced[d];
        const std::int64_t coord = remainder % dim;
        remainder /= dim;
        off += coord * plan.stride_in_reduced[d];
    }
    return in_base[off];
}

// ---------- Whole-tensor pass 1 -----------------------------------------
//
// Each block reads `kBlockSize` elements per stride step, folds them
// into a shared-memory tile, then performs a tree reduction down to a
// single block-level partial. The partial is written into
// `partials[blockIdx.x]`. The functor is injected as a template
// parameter so this kernel works for sum / prod / max / min.

template <class T, class Acc, class Op, class OutT>
__global__ void whole_tensor_pass1(const T* in_base, ops::ReductionPlan plan, Acc* partials) {
    extern __shared__ unsigned char smem_raw[];
    Acc* smem = reinterpret_cast<Acc*>(smem_raw);

    const int tid = threadIdx.x;
    const std::int64_t total = plan.reduced_numel;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

    Acc acc = Op::template identity<Acc>();
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + tid; i < total;
         i += stride) {
        const T v = load_at_linear<T>(in_base, plan, i);
        Op::template apply<Acc, T>(acc, v);
    }
    smem[tid] = acc;
    __syncthreads();

    // Tree reduction within the block.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            Op::template apply<Acc, Acc>(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = smem[0];
    }
}

// ---------- Whole-tensor pass 2 -----------------------------------------
//
// One block reduces the per-block partials produced by pass 1 into the
// 0-d output. `num_partials` is at most `kMaxPartials` (set at launch),
// so a single block always fits.

template <class Acc, class Op, class OutT>
__global__ void whole_tensor_pass2(const Acc* partials, std::int64_t num_partials, OutT* out_base,
                                   std::int64_t out_offset) {
    extern __shared__ unsigned char smem_raw[];
    Acc* smem = reinterpret_cast<Acc*>(smem_raw);

    const int tid = threadIdx.x;
    Acc acc = Op::template identity<Acc>();
    for (std::int64_t i = tid; i < num_partials; i += blockDim.x) {
        Op::template apply<Acc, Acc>(acc, partials[i]);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            Op::template apply<Acc, Acc>(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_base[out_offset] = static_cast<OutT>(smem[0]);
    }
}

// ---------- Launch helper ----------------------------------------------

class CudaScratchBuffer {
  public:
    CudaScratchBuffer(Device d, std::size_t bytes) : device_(d), bytes_(bytes) {
        if (bytes == 0) {
            return;
        }
        ptr_ = default_allocator(d)->allocate(bytes);
    }
    ~CudaScratchBuffer() {
        if (ptr_ != nullptr) {
            default_allocator(device_)->deallocate(ptr_, bytes_);
        }
    }
    CudaScratchBuffer(const CudaScratchBuffer&) = delete;
    CudaScratchBuffer& operator=(const CudaScratchBuffer&) = delete;
    CudaScratchBuffer(CudaScratchBuffer&&) = delete;
    CudaScratchBuffer& operator=(CudaScratchBuffer&&) = delete;
    void* get() const { return ptr_; }

  private:
    Device device_;
    std::size_t bytes_;
    void* ptr_ = nullptr;
};

template <class T, class Acc, class Op, class OutT>
void launch_whole_tensor(const Tensor& in, Tensor& out) {
    cuda::DeviceGuard device_guard(out.device().index);

    // Build a ReductionPlan with every input axis flagged as reduced —
    // matches what `canonicalise(x, {})` produced at the front-door.
    ops::ReductionAxes ax{};
    ax.rank = static_cast<int>(in.shape().size());
    ax.kept_numel = 1;
    ax.reduced_numel = in.numel();
    for (int d = 0; d < ax.rank; ++d) {
        ax.reduce[static_cast<std::size_t>(d)] = true;
    }
    const auto plan = ops::make_reduction_plan(in, out, ax);

    const std::int64_t n = plan.reduced_numel;
    const auto* in_base = static_cast<const T*>(in.storage().data());
    auto* out_base = static_cast<OutT*>(out.storage().data());

    // Tiny inputs go through a single-block path so the scratch buffer
    // never needs more than one slot. Mitigates issue #9 §7 risk-table
    // line 2 (mis-tuned grid for tiny inputs).
    constexpr int kMaxPartials = 1024;
    int blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > kMaxPartials) {
        blocks = kMaxPartials;
    }

    CudaScratchBuffer partials(out.device(), static_cast<std::size_t>(blocks) * sizeof(Acc));

    const std::size_t smem_bytes = static_cast<std::size_t>(kBlockSize) * sizeof(Acc);
    whole_tensor_pass1<T, Acc, Op, OutT>
        <<<blocks, kBlockSize, smem_bytes>>>(in_base, plan, static_cast<Acc*>(partials.get()));
    check_cuda("reduction_cuda_pass1");

    whole_tensor_pass2<Acc, Op, OutT><<<1, kBlockSize, smem_bytes>>>(
        static_cast<Acc*>(partials.get()), static_cast<std::int64_t>(blocks), out_base,
        plan.out_offset_elems);
    check_cuda("reduction_cuda_pass2");
}

// ---------- Axis-reduction kernel (multi-axis, values only) -----------

template <class T, class Acc, class Op, class OutT>
__global__ void axis_reduce_kernel(const T* in_base, OutT* out_base, ops::ReductionPlan plan) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= plan.kept_numel) {
        return;
    }
    // Decode the linear kept-axis index `k` into base offsets via the
    // kept odometer (input-side strides come from the plan).
    std::int64_t in_kept_off = plan.in_offset_elems;
    std::int64_t out_off = plan.out_offset_elems;
    std::int64_t remainder = k;
    for (int d = plan.rank_kept - 1; d >= 0; --d) {
        const std::int64_t dim = plan.shape_kept[d];
        const std::int64_t coord = remainder % dim;
        remainder /= dim;
        in_kept_off += coord * plan.stride_in_kept[d];
        out_off += coord * plan.stride_out[d];
    }

    Acc acc = Op::template identity<Acc>();
    // Per-thread reduced-axis odometer. Lives in registers (kMaxRank
    // entries); same advance pattern as the CPU run_reduction kernel.
    std::int64_t ridx[ops::kMaxRank] = {};
    std::int64_t in_red_off = 0;
    for (std::int64_t r = 0; r < plan.reduced_numel; ++r) {
        const T v = in_base[in_kept_off + in_red_off];
        Op::template apply<Acc, T>(acc, v);
        for (int d = plan.rank_reduced - 1; d >= 0; --d) {
            ++ridx[d];
            in_red_off += plan.stride_in_reduced[d];
            if (ridx[d] < plan.shape_reduced[d]) {
                break;
            }
            ridx[d] = 0;
            in_red_off -= plan.stride_in_reduced[d] * plan.shape_reduced[d];
        }
    }
    out_base[out_off] = static_cast<OutT>(acc);
}

template <class T, class Acc, class Op, class OutT>
void launch_axis_reduce(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    cuda::DeviceGuard device_guard(out.device().index);
    const auto plan = ops::make_reduction_plan(in, out, ax);
    if (plan.kept_numel == 0) {
        return;
    }
    const auto* in_base = static_cast<const T*>(in.storage().data());
    auto* out_base = static_cast<OutT*>(out.storage().data());
    const int blocks = static_cast<int>((plan.kept_numel + kBlockSize - 1) / kBlockSize);
    axis_reduce_kernel<T, Acc, Op, OutT><<<blocks, kBlockSize>>>(in_base, out_base, plan);
    check_cuda("reduction_cuda_axis");
}

// ---------- Axis-with-indices kernel (single reduced axis) ------------

template <class T, class Op, bool RecordValue>
__global__ void axis_with_idx_kernel(const T* in_base, T* vals_base, std::int64_t* idx_base,
                                     ops::ReductionPlan plan, std::int64_t reduced_size,
                                     std::int64_t reduced_stride) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= plan.kept_numel) {
        return;
    }
    std::int64_t in_kept_off = plan.in_offset_elems;
    std::int64_t out_off = plan.out_offset_elems;
    std::int64_t remainder = k;
    for (int d = plan.rank_kept - 1; d >= 0; --d) {
        const std::int64_t dim = plan.shape_kept[d];
        const std::int64_t coord = remainder % dim;
        remainder /= dim;
        in_kept_off += coord * plan.stride_in_kept[d];
        out_off += coord * plan.stride_out[d];
    }

    // Seed from r=0 so first-occurrence-wins applies even if the slice
    // is all-NaN (matches CPU + PyTorch semantics).
    T best = in_base[in_kept_off];
    std::int64_t best_idx = 0;
    for (std::int64_t r = 1; r < reduced_size; ++r) {
        const T v = in_base[in_kept_off + r * reduced_stride];
        if (Op::template should_replace<T>(best, v)) {
            best = v;
            best_idx = r;
        }
    }
    if constexpr (RecordValue) {
        vals_base[out_off] = best;
    }
    idx_base[out_off] = best_idx;
}

template <class T, class Op, bool RecordValue>
void launch_axis_with_idx(const Tensor& in, Tensor* vals_out, Tensor& idx_out, int axis) {
    cuda::DeviceGuard device_guard(idx_out.device().index);
    const auto& shape = in.shape();
    const auto& stride = in.stride();
    const int rank = static_cast<int>(shape.size());
    const std::int64_t reduced_size = shape[static_cast<std::size_t>(axis)];
    const std::int64_t reduced_stride = stride[static_cast<std::size_t>(axis)];

    ops::ReductionAxes ax{};
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
    if (plan.kept_numel == 0) {
        return;
    }

    const auto* in_base = static_cast<const T*>(in.storage().data());
    T* vals_base = vals_out != nullptr ? static_cast<T*>(vals_out->storage().data()) : nullptr;
    auto* idx_base = static_cast<std::int64_t*>(idx_out.storage().data());

    const int blocks = static_cast<int>((plan.kept_numel + kBlockSize - 1) / kBlockSize);
    axis_with_idx_kernel<T, Op, RecordValue>
        <<<blocks, kBlockSize>>>(in_base, vals_base, idx_base, plan, reduced_size, reduced_stride);
    check_cuda("reduction_cuda_axis_with_idx");
}

// ---------- Per-op dtype dispatch -------------------------------------

// Pick the right kernel family based on `ax`. Whole-tensor inputs go
// through the tree-reduction path (faster, used by the bench);
// everything else uses the per-output-element axis kernel.
template <class T, class Acc, class Op, class OutT>
void dispatch_reduce_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel == 1) {
        launch_whole_tensor<T, Acc, Op, OutT>(in, out);
    } else {
        launch_axis_reduce<T, Acc, Op, OutT>(in, out, ax);
    }
}

void sum_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    switch (in.dtype()) {
    case dtype::float32:
        dispatch_reduce_cuda<float, double, ops::SumF, float>(in, out, ax);
        break;
    case dtype::float64:
        dispatch_reduce_cuda<double, double, ops::SumF, double>(in, out, ax);
        break;
    case dtype::int32:
        dispatch_reduce_cuda<std::int32_t, std::int64_t, ops::SumF, std::int64_t>(in, out, ax);
        break;
    case dtype::int64:
        dispatch_reduce_cuda<std::int64_t, std::int64_t, ops::SumF, std::int64_t>(in, out, ax);
        break;
    case dtype::bool_:
        dispatch_reduce_cuda<unsigned char, std::int64_t, ops::SumF, std::int64_t>(in, out, ax);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::sum: bfloat16 reductions are not supported");
    }
}

void prod_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    switch (in.dtype()) {
    case dtype::float32:
        dispatch_reduce_cuda<float, double, ops::ProdF, float>(in, out, ax);
        break;
    case dtype::float64:
        dispatch_reduce_cuda<double, double, ops::ProdF, double>(in, out, ax);
        break;
    case dtype::int32:
        dispatch_reduce_cuda<std::int32_t, std::int64_t, ops::ProdF, std::int64_t>(in, out, ax);
        break;
    case dtype::int64:
        dispatch_reduce_cuda<std::int64_t, std::int64_t, ops::ProdF, std::int64_t>(in, out, ax);
        break;
    case dtype::bool_:
        dispatch_reduce_cuda<unsigned char, std::int64_t, ops::ProdF, std::int64_t>(in, out, ax);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::prod: bfloat16 reductions are not supported");
    }
}

template <class T> __global__ void mean_finalize(T* out, std::int64_t out_offset, T inv) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[out_offset] *= inv;
    }
}

template <class T> __global__ void mean_finalize_nan(T* out, std::int64_t out_offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Portable NaN that does not depend on libstdc++ <limits> being
        // constexpr-callable from device code. PyTorch's empty-slice
        // mean returns NaN; we mirror that here.
        const T zero = static_cast<T>(0);
        out[out_offset] = zero / zero;
    }
}

// Finalisation kernel for the axis path: divides every output by
// `reduced_numel` (or writes NaN when the slice was empty). Same
// contract as `mean_finalize` but operates on the whole output buffer.
template <class T>
__global__ void mean_axis_finalize(T* out_base, std::int64_t out_offset, std::int64_t out_numel,
                                   T inv) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= out_numel) {
        return;
    }
    out_base[out_offset + i] *= inv;
}

template <class T>
__global__ void mean_axis_finalize_nan(T* out_base, std::int64_t out_offset,
                                       std::int64_t out_numel) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= out_numel) {
        return;
    }
    const T zero = static_cast<T>(0);
    out_base[out_offset + i] = zero / zero;
}

template <class T> void launch_mean(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel == 1) {
        launch_whole_tensor<T, double, ops::SumF, T>(in, out);
    } else {
        launch_axis_reduce<T, double, ops::SumF, T>(in, out, ax);
    }
    cuda::DeviceGuard device_guard(out.device().index);
    auto* out_base = static_cast<T*>(out.storage().data());
    const std::int64_t out_numel = out.numel();
    if (out_numel == 0) {
        return;
    }
    const int blocks = static_cast<int>((out_numel + kBlockSize - 1) / kBlockSize);
    if (ax.reduced_numel == 0) {
        mean_axis_finalize_nan<T><<<blocks, kBlockSize>>>(out_base, out.offset(), out_numel);
    } else {
        const T inv = static_cast<T>(1) / static_cast<T>(ax.reduced_numel);
        mean_axis_finalize<T><<<blocks, kBlockSize>>>(out_base, out.offset(), out_numel, inv);
    }
    check_cuda("mean_finalize");
}

void mean_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    switch (in.dtype()) {
    case dtype::float32:
        launch_mean<float>(in, out, ax);
        break;
    case dtype::float64:
        launch_mean<double>(in, out, ax);
        break;
    case dtype::int32:
    case dtype::int64:
    case dtype::bool_:
        throw DTypeError("ctorch::mean: requires a floating dtype input");
    case dtype::bfloat16:
        throw DTypeError("ctorch::mean: bfloat16 reductions are not supported");
    }
}

template <class Op>
void maxmin_value_cuda_dispatch(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax,
                                const char* name) {
    switch (in.dtype()) {
    case dtype::float32:
        dispatch_reduce_cuda<float, float, Op, float>(in, out, ax);
        break;
    case dtype::float64:
        dispatch_reduce_cuda<double, double, Op, double>(in, out, ax);
        break;
    case dtype::int32:
        dispatch_reduce_cuda<std::int32_t, std::int32_t, Op, std::int32_t>(in, out, ax);
        break;
    case dtype::int64:
        dispatch_reduce_cuda<std::int64_t, std::int64_t, Op, std::int64_t>(in, out, ax);
        break;
    case dtype::bool_:
        dispatch_reduce_cuda<unsigned char, unsigned char, Op, unsigned char>(in, out, ax);
        break;
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 reductions are not "
                         "supported");
    }
}

void max_val_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    maxmin_value_cuda_dispatch<ops::MaxF>(in, out, ax, "max");
}
void min_val_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    maxmin_value_cuda_dispatch<ops::MinF>(in, out, ax, "min");
}

template <class Op>
void maxmin_with_idx_cuda_dispatch(const Tensor& in, Tensor& vals, Tensor& idx, int axis,
                                   const char* name) {
    switch (in.dtype()) {
    case dtype::float32:
        launch_axis_with_idx<float, Op, true>(in, &vals, idx, axis);
        break;
    case dtype::float64:
        launch_axis_with_idx<double, Op, true>(in, &vals, idx, axis);
        break;
    case dtype::int32:
        launch_axis_with_idx<std::int32_t, Op, true>(in, &vals, idx, axis);
        break;
    case dtype::int64:
        launch_axis_with_idx<std::int64_t, Op, true>(in, &vals, idx, axis);
        break;
    case dtype::bool_:
        launch_axis_with_idx<unsigned char, Op, true>(in, &vals, idx, axis);
        break;
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 reductions are not "
                         "supported");
    }
}

void max_val_idx_cuda(const Tensor& in, Tensor& vals, Tensor& idx, int axis) {
    maxmin_with_idx_cuda_dispatch<ops::MaxF>(in, vals, idx, axis, "max");
}
void min_val_idx_cuda(const Tensor& in, Tensor& vals, Tensor& idx, int axis) {
    maxmin_with_idx_cuda_dispatch<ops::MinF>(in, vals, idx, axis, "min");
}

template <class Op>
void argmaxmin_cuda_dispatch(const Tensor& in, Tensor& idx, int axis, const char* name) {
    switch (in.dtype()) {
    case dtype::float32:
        launch_axis_with_idx<float, Op, false>(in, nullptr, idx, axis);
        break;
    case dtype::float64:
        launch_axis_with_idx<double, Op, false>(in, nullptr, idx, axis);
        break;
    case dtype::int32:
        launch_axis_with_idx<std::int32_t, Op, false>(in, nullptr, idx, axis);
        break;
    case dtype::int64:
        launch_axis_with_idx<std::int64_t, Op, false>(in, nullptr, idx, axis);
        break;
    case dtype::bool_:
        launch_axis_with_idx<unsigned char, Op, false>(in, nullptr, idx, axis);
        break;
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 reductions are not "
                         "supported");
    }
}

void argmax_cuda(const Tensor& in, Tensor& idx, int axis) {
    argmaxmin_cuda_dispatch<ops::MaxF>(in, idx, axis, "argmax");
}
void argmin_cuda(const Tensor& in, Tensor& idx, int axis) {
    argmaxmin_cuda_dispatch<ops::MinF>(in, idx, axis, "argmin");
}

} // namespace

extern "C" void ctorch_register_cuda_reduction_ops() {
    dispatch::register_op<op::SumOp>(Device::Kind::CUDA, &sum_cuda);
    dispatch::register_op<op::ProdOp>(Device::Kind::CUDA, &prod_cuda);
    dispatch::register_op<op::MeanOp>(Device::Kind::CUDA, &mean_cuda);
    dispatch::register_op<op::MaxValOp>(Device::Kind::CUDA, &max_val_cuda);
    dispatch::register_op<op::MinValOp>(Device::Kind::CUDA, &min_val_cuda);
    dispatch::register_op<op::MaxValIdxOp>(Device::Kind::CUDA, &max_val_idx_cuda);
    dispatch::register_op<op::MinValIdxOp>(Device::Kind::CUDA, &min_val_idx_cuda);
    dispatch::register_op<op::ArgmaxOp>(Device::Kind::CUDA, &argmax_cuda);
    dispatch::register_op<op::ArgminOp>(Device::Kind::CUDA, &argmin_cuda);
}

} // namespace ctorch
