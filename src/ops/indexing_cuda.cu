//===- src/ops/indexing_cuda.cu - CUDA index_select kernel ----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA gather kernel for `ctorch::index_select`. A small validation
/// kernel runs first: one thread per input index, normalises negative
/// values, range-checks against `src.shape[dim]`, and writes the
/// pre-multiplied source-axis offset into a device scratch buffer. If
/// any thread observed an out-of-range index it raises a flag, which is
/// copied back to the host and turned into a `ShapeError`. The gather
/// kernel then walks the contiguous output one thread per element,
/// reusing the scratch offsets so the per-thread bounds-check disappears.
///
//===----------------------------------------------------------------------===//

#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "cuda/device_guard.h"
#include "ops/tensor_iter.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
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

struct IndexSelectPlan {
    int rank = 0;
    int dim = 0;
    std::array<std::int64_t, ops::kMaxRank> out_shape{};
    std::array<std::int64_t, ops::kMaxRank> src_stride{};
    std::int64_t src_offset_elems = 0;
    std::int64_t out_offset_elems = 0;
    std::int64_t total_out = 0;
};

template <class I>
__global__ void validate_indices_kernel(const I* idx_base, std::int64_t idx_stride_elems,
                                        std::int64_t n_indices, std::int64_t src_dim_size,
                                        std::int64_t src_dim_stride, std::int64_t* resolved,
                                        int* err_flag) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_indices) {
        return;
    }
    std::int64_t v = static_cast<std::int64_t>(idx_base[i * idx_stride_elems]);
    if (v < 0) {
        v += src_dim_size;
    }
    if (v < 0 || v >= src_dim_size) {
        atomicMax(err_flag, 1);
        resolved[i] = 0;
    } else {
        resolved[i] = v * src_dim_stride;
    }
}

template <class T>
__global__ void index_select_gather_kernel(const T* src_base, T* out_base,
                                           const std::int64_t* resolved, IndexSelectPlan plan) {
    const std::int64_t lin = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lin >= plan.total_out) {
        return;
    }
    std::int64_t src_off = plan.src_offset_elems;
    std::int64_t remainder = lin;
    for (int i = plan.rank - 1; i >= 0; --i) {
        const std::int64_t dim_size = plan.out_shape[i];
        const std::int64_t coord = remainder % dim_size;
        remainder /= dim_size;
        if (i == plan.dim) {
            src_off += resolved[coord];
        } else {
            src_off += coord * plan.src_stride[i];
        }
    }
    // Output is contiguous; lin is the offset within the output buffer.
    out_base[plan.out_offset_elems + lin] = src_base[src_off];
}

template <class T, class I>
void launch_index_select(const Tensor& src, int dim, const Tensor& indices, Tensor& out) {
    cuda::DeviceGuard device_guard(out.device().index);

    const auto& src_shape = src.shape();
    const auto& src_stride = src.stride();
    if (src_shape.size() > ops::kMaxRank) {
        // The kernel passes IndexSelectPlan by value; its fixed-size arrays
        // can't accommodate ranks above kMaxRank. The CPU backend still
        // works for arbitrary rank, so this guard lives only on the CUDA
        // path (matches the rest of the CUDA op family).
        throw ShapeError("ctorch::index_select: CUDA backend does not support tensor rank > " +
                         std::to_string(ops::kMaxRank) + " (got " +
                         std::to_string(src_shape.size()) + ")");
    }
    const int rank = static_cast<int>(src_shape.size());
    const std::int64_t src_dim_size = src_shape[static_cast<std::size_t>(dim)];
    const std::int64_t src_dim_stride = src_stride[static_cast<std::size_t>(dim)];
    const std::int64_t n_indices = indices.numel();

    IndexSelectPlan plan{};
    plan.rank = rank;
    plan.dim = dim;
    plan.src_offset_elems = src.offset();
    plan.out_offset_elems = out.offset();
    plan.total_out = out.numel();
    for (int i = 0; i < rank; ++i) {
        plan.out_shape[static_cast<std::size_t>(i)] = out.shape()[static_cast<std::size_t>(i)];
        plan.src_stride[static_cast<std::size_t>(i)] = src_stride[static_cast<std::size_t>(i)];
    }

    if (n_indices == 0) {
        // Nothing to validate or gather.
        return;
    }

    // Scratch: 1×int err flag + n_indices×int64 resolved offsets.
    const std::size_t flag_bytes = sizeof(int);
    const std::size_t resolved_bytes = static_cast<std::size_t>(n_indices) * sizeof(std::int64_t);
    CudaScratchBuffer flag_buf(out.device(), flag_bytes);
    CudaScratchBuffer resolved_buf(out.device(), resolved_bytes);

    if (cudaMemset(flag_buf.get(), 0, flag_bytes) != cudaSuccess) {
        throw DeviceError("ctorch::index_select: cudaMemset failed");
    }

    const std::int64_t idx_stride_elems = indices.stride().empty() ? 0 : indices.stride()[0];
    const auto* idx_base = static_cast<const I*>(indices.storage().data()) + indices.offset();

    const int validate_blocks = static_cast<int>((n_indices + kBlockSize - 1) / kBlockSize);
    validate_indices_kernel<I><<<validate_blocks, kBlockSize>>>(
        idx_base, idx_stride_elems, n_indices, src_dim_size, src_dim_stride,
        static_cast<std::int64_t*>(resolved_buf.get()), static_cast<int*>(flag_buf.get()));
    check_cuda("index_select_validate");

    int err_flag_host = 0;
    if (cudaMemcpy(&err_flag_host, flag_buf.get(), flag_bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        throw DeviceError("ctorch::index_select: cudaMemcpy of err flag failed");
    }
    if (err_flag_host != 0) {
        throw ShapeError("ctorch::index_select: index value out of range for dim " +
                         std::to_string(dim) + " of size " + std::to_string(src_dim_size));
    }

    if (plan.total_out == 0) {
        return;
    }

    const auto* src_base = static_cast<const T*>(src.storage().data());
    auto* out_base = static_cast<T*>(out.storage().data());
    const int gather_blocks = static_cast<int>((plan.total_out + kBlockSize - 1) / kBlockSize);
    index_select_gather_kernel<T><<<gather_blocks, kBlockSize>>>(
        src_base, out_base, static_cast<const std::int64_t*>(resolved_buf.get()), plan);
    check_cuda("index_select_gather");
}

template <class T>
void index_select_dispatch_idx(const Tensor& src, int dim, const Tensor& indices, Tensor& out) {
    switch (indices.dtype()) {
    case dtype::int32:
        launch_index_select<T, std::int32_t>(src, dim, indices, out);
        break;
    case dtype::int64:
        launch_index_select<T, std::int64_t>(src, dim, indices, out);
        break;
    default:
        throw DTypeError("ctorch::index_select: indices dtype must be int32 or int64");
    }
}

void index_select_cuda(const Tensor& src, int dim, const Tensor& indices, Tensor& out) {
    switch (src.dtype()) {
    case dtype::float32:
        index_select_dispatch_idx<float>(src, dim, indices, out);
        break;
    case dtype::float64:
        index_select_dispatch_idx<double>(src, dim, indices, out);
        break;
    case dtype::int32:
        index_select_dispatch_idx<std::int32_t>(src, dim, indices, out);
        break;
    case dtype::int64:
        index_select_dispatch_idx<std::int64_t>(src, dim, indices, out);
        break;
    case dtype::bool_:
        index_select_dispatch_idx<unsigned char>(src, dim, indices, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch::index_select: bfloat16 is not supported");
    }
}

} // namespace

extern "C" void ctorch_register_cuda_indexing_ops() {
    dispatch::register_op<op::IndexSelectOp>(Device::Kind::CUDA, &index_select_cuda);
}

} // namespace ctorch
