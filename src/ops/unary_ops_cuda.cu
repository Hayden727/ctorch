//===- src/ops/unary_ops_cuda.cu - CUDA element-wise unary ops -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CUDA kernels for neg/abs/exp/log/sqrt/relu/sigmoid/tanh.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "ops/functors.h"
#include "ops/tensor_iter.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

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
__global__ void unary_kernel_contig(T* __restrict__ out,
                                    const T* __restrict__ in,
                                    std::int64_t n, Op op) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = op(in[i]);
    }
}

template <class T, class Op>
__global__ void unary_kernel_strided(T* out_base, const T* in_base,
                                     ops::UnaryIndexer ctx, Op op) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= ctx.n) {
        return;
    }
    std::int64_t remainder = i;
    std::int64_t in_off = ctx.in_offset_elems;
    std::int64_t out_off = ctx.out_offset_elems;
    for (int d = ctx.rank - 1; d >= 0; --d) {
        const std::int64_t coord = remainder % ctx.shape[d];
        remainder /= ctx.shape[d];
        in_off += coord * ctx.in_stride[d];
        out_off += coord * ctx.out_stride[d];
    }
    out_base[out_off] = op(in_base[in_off]);
}

template <class T, class Op>
void launch_unary(const Tensor& in, Tensor& out, Op op) {
    if (ops::can_use_contiguous_path(in, out)) {
        const std::int64_t n = out.numel();
        if (n == 0) {
            return; // empty tensor: nothing to launch (zero-block grids are invalid)
        }
        const auto* ip = static_cast<const T*>(in.storage().data()) + in.offset();
        auto* op_out = static_cast<T*>(out.storage().data()) + out.offset();
        const int blocks = blocks_for(n);
        unary_kernel_contig<T, Op><<<blocks, kBlockSize>>>(op_out, ip, n, op);
    } else {
        const auto ctx = ops::make_unary_indexer(in, out);
        if (ctx.n == 0) {
            return;
        }
        const auto* in_base = static_cast<const T*>(in.storage().data());
        auto* out_base = static_cast<T*>(out.storage().data());
        const int blocks = blocks_for(ctx.n);
        unary_kernel_strided<T, Op><<<blocks, kBlockSize>>>(out_base, in_base, ctx, op);
    }
    check_cuda("unary_op_cuda");
}

template <class Op>
void unary_dispatch_signed_numeric(const Tensor& in, Tensor& out, Op op,
                                   const char* name) {
    switch (out.dtype()) {
    case dtype::float32:
        launch_unary<float>(in, out, op);
        break;
    case dtype::float64:
        launch_unary<double>(in, out, op);
        break;
    case dtype::int32:
        launch_unary<std::int32_t>(in, out, op);
        break;
    case dtype::int64:
        launch_unary<std::int64_t>(in, out, op);
        break;
    case dtype::bool_:
        throw DTypeError(std::string("ctorch::") + name +
                         ": not defined on bool — promote to int32 first");
    case dtype::bfloat16:
        throw DTypeError(std::string("ctorch::") + name +
                         ": bfloat16 not supported in Issue 03");
    }
}

template <class Op>
void unary_dispatch_float(const Tensor& in, Tensor& out, Op op, const char* name) {
    switch (out.dtype()) {
    case dtype::float32:
        launch_unary<float>(in, out, op);
        break;
    case dtype::float64:
        launch_unary<double>(in, out, op);
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

void neg_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_signed_numeric(in, out, ops::NegF{}, "neg");
}
void abs_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_signed_numeric(in, out, ops::AbsF{}, "abs");
}
void relu_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_signed_numeric(in, out, ops::ReluF{}, "relu");
}
void exp_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::ExpF{}, "exp");
}
void log_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::LogF{}, "log");
}
void sqrt_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::SqrtF{}, "sqrt");
}
void sigmoid_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::SigmoidF{}, "sigmoid");
}
void tanh_cuda(const Tensor& in, Tensor& out) {
    unary_dispatch_float(in, out, ops::TanhF{}, "tanh");
}

struct CUDAUnaryRegistrar {
    CUDAUnaryRegistrar() {
        dispatch::register_op<op::NegOp>(Device::Kind::CUDA, &neg_cuda);
        dispatch::register_op<op::AbsOp>(Device::Kind::CUDA, &abs_cuda);
        dispatch::register_op<op::ReluOp>(Device::Kind::CUDA, &relu_cuda);
        dispatch::register_op<op::ExpOp>(Device::Kind::CUDA, &exp_cuda);
        dispatch::register_op<op::LogOp>(Device::Kind::CUDA, &log_cuda);
        dispatch::register_op<op::SqrtOp>(Device::Kind::CUDA, &sqrt_cuda);
        dispatch::register_op<op::SigmoidOp>(Device::Kind::CUDA, &sigmoid_cuda);
        dispatch::register_op<op::TanhOp>(Device::Kind::CUDA, &tanh_cuda);
    }
};
const CUDAUnaryRegistrar kCudaUnaryRegistrar{};

} // namespace

} // namespace ctorch
