//===- src/ops/binary_ops_cuda.cu - CUDA element-wise binary ops *- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA kernels for add/sub/mul/div (free + in-place). One templated
/// elementwise kernel handles every binary op; the op is injected as a
/// __host__ __device__ functor from src/ops/functors.h so the arithmetic
/// is byte-identical to the CPU path.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "ops/broadcast.h"
#include "ops/functors.h"
#include "ops/tensor_iter.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ctorch {

namespace {

constexpr int kBlockSize = 256;

inline int blocks_for(std::int64_t n) {
    return static_cast<int>((n + kBlockSize - 1) / kBlockSize);
}

void check_cuda(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw DeviceError(std::string("ctorch::") + what + ": CUDA error: " +
                          cudaGetErrorString(err));
    }
}

// ---- Templated kernels ---------------------------------------------------

template <class T, class Op>
__global__ void binary_kernel_contig(T* __restrict__ out,
                                     const T* __restrict__ a,
                                     const T* __restrict__ b,
                                     std::int64_t n, Op op) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = op(a[i], b[i]);
    }
}

template <class T, class Op>
__global__ void binary_kernel_strided(T* out_base, const T* a_base, const T* b_base,
                                      ops::BinaryIndexer ctx, Op op) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= ctx.n) {
        return;
    }
    std::int64_t remainder = i;
    std::int64_t a_off = ctx.a_offset_elems;
    std::int64_t b_off = ctx.b_offset_elems;
    std::int64_t out_off = ctx.out_offset_elems;
    for (int d = ctx.rank - 1; d >= 0; --d) {
        const std::int64_t coord = remainder % ctx.shape[d];
        remainder /= ctx.shape[d];
        a_off += coord * ctx.a_stride[d];
        b_off += coord * ctx.b_stride[d];
        out_off += coord * ctx.out_stride[d];
    }
    out_base[out_off] = op(a_base[a_off], b_base[b_off]);
}

template <class T, class Op>
void launch_binary(const Tensor& a, const Tensor& b, Tensor& out, Op op) {
    if (ops::can_use_contiguous_path(a, b, out)) {
        const std::int64_t n = out.numel();
        if (n == 0) {
            return; // empty tensor: nothing to launch (zero-block grids are invalid)
        }
        const auto* ap = static_cast<const T*>(a.storage().data()) + a.offset();
        const auto* bp = static_cast<const T*>(b.storage().data()) + b.offset();
        auto* op_out = static_cast<T*>(out.storage().data()) + out.offset();
        const int blocks = blocks_for(n);
        binary_kernel_contig<T, Op><<<blocks, kBlockSize>>>(op_out, ap, bp, n, op);
    } else {
        const auto br = ops::broadcast_two(a, b);
        const auto ctx = ops::make_binary_indexer(a, b, out, br);
        if (ctx.n == 0) {
            return;
        }
        const auto* a_base = static_cast<const T*>(a.storage().data());
        const auto* b_base = static_cast<const T*>(b.storage().data());
        auto* out_base = static_cast<T*>(out.storage().data());
        const int blocks = blocks_for(ctx.n);
        binary_kernel_strided<T, Op><<<blocks, kBlockSize>>>(out_base, a_base, b_base, ctx, op);
    }
    check_cuda("binary_op_cuda");
}

template <class Op>
void binary_dispatch_numeric(const Tensor& a, const Tensor& b, Tensor& out, Op op) {
    switch (out.dtype()) {
    case dtype::float32:
        launch_binary<float>(a, b, out, op);
        break;
    case dtype::float64:
        launch_binary<double>(a, b, out, op);
        break;
    case dtype::int32:
        launch_binary<std::int32_t>(a, b, out, op);
        break;
    case dtype::int64:
        launch_binary<std::int64_t>(a, b, out, op);
        break;
    case dtype::bool_:
        throw DTypeError("ctorch: binary arithmetic on bool is not supported "
                         "(promote to int32 first)");
    case dtype::bfloat16:
        throw DTypeError("ctorch: bfloat16 not supported in Issue 03");
    }
}

// ---- Backend functions registered with the dispatcher --------------------

void add_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::AddF{});
}
void sub_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::SubF{});
}
void mul_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::MulF{});
}
void div_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    binary_dispatch_numeric(a, b, out, ops::DivF{});
}

void add_inplace_cuda(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::AddF{});
}
void sub_inplace_cuda(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::SubF{});
}
void mul_inplace_cuda(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::MulF{});
}
void div_inplace_cuda(Tensor& a, const Tensor& b) {
    binary_dispatch_numeric(a, b, a, ops::DivF{});
}

struct CUDABinaryRegistrar {
    CUDABinaryRegistrar() {
        dispatch::register_op<op::AddOp>(Device::Kind::CUDA, &add_cuda);
        dispatch::register_op<op::SubOp>(Device::Kind::CUDA, &sub_cuda);
        dispatch::register_op<op::MulOp>(Device::Kind::CUDA, &mul_cuda);
        dispatch::register_op<op::DivOp>(Device::Kind::CUDA, &div_cuda);
        dispatch::register_op<op::AddInplaceOp>(Device::Kind::CUDA, &add_inplace_cuda);
        dispatch::register_op<op::SubInplaceOp>(Device::Kind::CUDA, &sub_inplace_cuda);
        dispatch::register_op<op::MulInplaceOp>(Device::Kind::CUDA, &mul_inplace_cuda);
        dispatch::register_op<op::DivInplaceOp>(Device::Kind::CUDA, &div_inplace_cuda);
    }
};
const CUDABinaryRegistrar kCudaBinaryRegistrar{};

} // namespace

} // namespace ctorch
