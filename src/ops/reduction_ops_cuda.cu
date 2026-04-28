//===- src/ops/reduction_ops_cuda.cu - CUDA reduction kernels --*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA kernels for reductions. This commit ships the whole-tensor
/// path — a two-pass tree reduction that handles any input rank or
/// stride pattern by walking the reduced subspace via the shared
/// `ReductionPlan` odometer. Per-tile partials accumulate in shared
/// memory; a single block reduces the partials into the 0-d output.
///
/// Axis reductions (kept_numel > 1) are stubbed in this commit and
/// throw `DeviceError`; commit 5 fills in the kernel.
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

// ---------- Per-op dtype dispatch + axis-path stub --------------------

[[noreturn]] void axis_not_implemented_yet(const char* name) {
    throw DeviceError(std::string("ctorch::") + name +
                      ": axis reductions on CUDA are not yet implemented "
                      "(landing in commit 5)");
}

void sum_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel != 1) {
        axis_not_implemented_yet("sum");
    }
    switch (in.dtype()) {
    case dtype::float32:
        launch_whole_tensor<float, double, ops::SumF, float>(in, out);
        break;
    case dtype::float64:
        launch_whole_tensor<double, double, ops::SumF, double>(in, out);
        break;
    case dtype::int32:
        launch_whole_tensor<std::int32_t, std::int64_t, ops::SumF, std::int64_t>(in, out);
        break;
    case dtype::int64:
        launch_whole_tensor<std::int64_t, std::int64_t, ops::SumF, std::int64_t>(in, out);
        break;
    case dtype::bool_:
        launch_whole_tensor<unsigned char, std::int64_t, ops::SumF, std::int64_t>(in, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::sum: bfloat16 reductions are not supported");
    }
}

void prod_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel != 1) {
        axis_not_implemented_yet("prod");
    }
    switch (in.dtype()) {
    case dtype::float32:
        launch_whole_tensor<float, double, ops::ProdF, float>(in, out);
        break;
    case dtype::float64:
        launch_whole_tensor<double, double, ops::ProdF, double>(in, out);
        break;
    case dtype::int32:
        launch_whole_tensor<std::int32_t, std::int64_t, ops::ProdF, std::int64_t>(in, out);
        break;
    case dtype::int64:
        launch_whole_tensor<std::int64_t, std::int64_t, ops::ProdF, std::int64_t>(in, out);
        break;
    case dtype::bool_:
        launch_whole_tensor<unsigned char, std::int64_t, ops::ProdF, std::int64_t>(in, out);
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

template <class T> void launch_mean_whole_tensor(const Tensor& in, Tensor& out) {
    launch_whole_tensor<T, double, ops::SumF, T>(in, out);
    cuda::DeviceGuard device_guard(out.device().index);
    auto* out_base = static_cast<T*>(out.storage().data());
    const std::int64_t reduced = in.numel();
    if (reduced == 0) {
        mean_finalize_nan<T><<<1, 1>>>(out_base, out.offset());
    } else {
        const T inv = static_cast<T>(1) / static_cast<T>(reduced);
        mean_finalize<T><<<1, 1>>>(out_base, out.offset(), inv);
    }
    check_cuda("mean_finalize");
}

void mean_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel != 1) {
        axis_not_implemented_yet("mean");
    }
    switch (in.dtype()) {
    case dtype::float32:
        launch_mean_whole_tensor<float>(in, out);
        break;
    case dtype::float64:
        launch_mean_whole_tensor<double>(in, out);
        break;
    case dtype::int32:
    case dtype::int64:
    case dtype::bool_:
        throw DTypeError("ctorch::mean: requires a floating dtype input");
    case dtype::bfloat16:
        throw DTypeError("ctorch::mean: bfloat16 reductions are not supported");
    }
}

void max_val_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel != 1) {
        axis_not_implemented_yet("max");
    }
    switch (in.dtype()) {
    case dtype::float32:
        launch_whole_tensor<float, float, ops::MaxF, float>(in, out);
        break;
    case dtype::float64:
        launch_whole_tensor<double, double, ops::MaxF, double>(in, out);
        break;
    case dtype::int32:
        launch_whole_tensor<std::int32_t, std::int32_t, ops::MaxF, std::int32_t>(in, out);
        break;
    case dtype::int64:
        launch_whole_tensor<std::int64_t, std::int64_t, ops::MaxF, std::int64_t>(in, out);
        break;
    case dtype::bool_:
        launch_whole_tensor<unsigned char, unsigned char, ops::MaxF, unsigned char>(in, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::max: bfloat16 reductions are not supported");
    }
}

void min_val_cuda(const Tensor& in, Tensor& out, const ops::ReductionAxes& ax) {
    if (ax.kept_numel != 1) {
        axis_not_implemented_yet("min");
    }
    switch (in.dtype()) {
    case dtype::float32:
        launch_whole_tensor<float, float, ops::MinF, float>(in, out);
        break;
    case dtype::float64:
        launch_whole_tensor<double, double, ops::MinF, double>(in, out);
        break;
    case dtype::int32:
        launch_whole_tensor<std::int32_t, std::int32_t, ops::MinF, std::int32_t>(in, out);
        break;
    case dtype::int64:
        launch_whole_tensor<std::int64_t, std::int64_t, ops::MinF, std::int64_t>(in, out);
        break;
    case dtype::bool_:
        launch_whole_tensor<unsigned char, unsigned char, ops::MinF, unsigned char>(in, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::min: bfloat16 reductions are not supported");
    }
}

} // namespace

extern "C" void ctorch_register_cuda_reduction_ops() {
    dispatch::register_op<op::SumOp>(Device::Kind::CUDA, &sum_cuda);
    dispatch::register_op<op::ProdOp>(Device::Kind::CUDA, &prod_cuda);
    dispatch::register_op<op::MeanOp>(Device::Kind::CUDA, &mean_cuda);
    dispatch::register_op<op::MaxValOp>(Device::Kind::CUDA, &max_val_cuda);
    dispatch::register_op<op::MinValOp>(Device::Kind::CUDA, &min_val_cuda);
    // MaxValIdxOp / MinValIdxOp / ArgmaxOp / ArgminOp are wired in
    // commit 5 alongside the axis-reduction kernels.
}

} // namespace ctorch
