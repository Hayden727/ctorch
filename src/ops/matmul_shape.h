//===- src/ops/matmul_shape.h - matmul shape planning ---------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal helper that turns the five PyTorch matmul shape cases
/// (1-D × 1-D, 1-D × 2-D, 2-D × 1-D, 2-D × 2-D, batched N-D) into a
/// uniform `MatmulPlan` consumed by both the CPU and CUDA backends.
///
/// The plan is a list of `(a_offset, b_offset, c_offset)` triples in
/// **elements** (not bytes), one per batch step. Each triple selects a
/// single `(M × K) * (K × N)` GEMM the backend has to issue. Inputs
/// are assumed to be contiguous — the front-door materialises non-
/// contiguous operands before constructing the plan.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_MATMUL_SHAPE_H
#define CTORCH_OPS_MATMUL_SHAPE_H

#include "ctorch/tensor.h"

#include <cstdint>
#include <vector>

namespace ctorch::ops {

struct MatmulPlan {
    /// User-visible shape of the matmul result. Used by the front-door
    /// to allocate the output tensor.
    std::vector<std::int64_t> out_shape;

    /// Inner GEMM dimensions: A is `M × K`, B is `K × N`, C is `M × N`.
    std::int64_t M = 0;
    std::int64_t K = 0;
    std::int64_t N = 0;

    /// Per-batch offsets in **elements**. `batch_offsets[i]` is the
    /// starting element offset into the contiguous storage of A / B / C
    /// for batch step `i`. `size()` equals the total broadcasted batch
    /// count (1 for the non-batched cases).
    std::vector<std::int64_t> a_offsets;
    std::vector<std::int64_t> b_offsets;
    std::vector<std::int64_t> c_offsets;
};

/// Builds a `MatmulPlan` for the given inputs. Validates shape
/// compatibility (inner-dim mismatch raises `ShapeError`). Inputs may
/// have any rank; the plan handles 1-D promotion / 1-D squeezing
/// internally and returns the user-visible output shape.
MatmulPlan plan_matmul(const Tensor& a, const Tensor& b);

/// Backend entry points. Defined in `matmul_cpu.cpp` /
/// `matmul_cuda.cu`; referenced from `matmul.cpp`'s registrar so the
/// linker keeps both backend TUs alive when consumers link against
/// `ctorch_core` as a static library. (Static initialisers in lone
/// TUs would otherwise be discarded.)
void matmul_cpu(const Tensor& a, const Tensor& b, Tensor& out);
#if defined(CTORCH_HAS_CUDA)
void matmul_cuda(const Tensor& a, const Tensor& b, Tensor& out);
#endif

} // namespace ctorch::ops

#endif // CTORCH_OPS_MATMUL_SHAPE_H
