//===- src/ops/matmul_shape.cpp - matmul shape planning -------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ops/matmul_shape.h"

#include "ctorch/errors.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ctorch::ops {

namespace {

std::string shape_to_string(const std::vector<std::int64_t>& s) {
    std::string out = "(";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) {
            out += ", ";
        }
        out += std::to_string(s[i]);
    }
    out += ")";
    return out;
}

// Right-aligned broadcast of two batch shapes. Throws on mismatch.
std::vector<std::int64_t> broadcast_batch(const std::vector<std::int64_t>& a,
                                          const std::vector<std::int64_t>& b) {
    const std::size_t out_rank = std::max(a.size(), b.size());
    std::vector<std::int64_t> out(out_rank, 1);
    for (std::size_t i = 0; i < out_rank; ++i) {
        const std::int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
        const std::int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (da == db || da == 1 || db == 1) {
            out[out_rank - 1 - i] = std::max(da, db);
        } else {
            throw ShapeError("ctorch::matmul: incompatible batch dims " + std::to_string(da) +
                             " vs " + std::to_string(db));
        }
    }
    return out;
}

} // namespace

MatmulPlan plan_matmul(const Tensor& a, const Tensor& b) {
    const auto& as = a.shape();
    const auto& bs = b.shape();
    if (as.empty() || bs.empty()) {
        throw ShapeError("ctorch::matmul: 0-d tensors are not supported (got " +
                         shape_to_string(as) + " and " + shape_to_string(bs) + ")");
    }

    const bool a_is_1d = as.size() == 1;
    const bool b_is_1d = bs.size() == 1;

    // Promote 1-D to 2-D for the inner GEMM, remember to drop the
    // promoted axis from the user-visible output.
    std::vector<std::int64_t> a_eff = as;
    std::vector<std::int64_t> b_eff = bs;
    if (a_is_1d) {
        a_eff.insert(a_eff.begin(), 1); // [K] -> [1, K]
    }
    if (b_is_1d) {
        b_eff.push_back(1); // [K] -> [K, 1]
    }

    const std::int64_t M = a_eff[a_eff.size() - 2];
    const std::int64_t Ka = a_eff[a_eff.size() - 1];
    const std::int64_t Kb = b_eff[b_eff.size() - 2];
    const std::int64_t N = b_eff[b_eff.size() - 1];
    if (Ka != Kb) {
        throw ShapeError("ctorch::matmul: inner dimensions do not match — lhs " +
                         shape_to_string(as) + ", rhs " + shape_to_string(bs));
    }

    // Batch dims are everything except the trailing two.
    std::vector<std::int64_t> a_batch(a_eff.begin(), a_eff.end() - 2);
    std::vector<std::int64_t> b_batch(b_eff.begin(), b_eff.end() - 2);
    std::vector<std::int64_t> bcast = broadcast_batch(a_batch, b_batch);

    MatmulPlan plan;
    plan.M = M;
    plan.K = Ka;
    plan.N = N;

    // Per-batch element offsets, accounting for size-1 broadcasts (those
    // axes contribute stride 0). Strides walk the batch portion of the
    // contiguous input layout.
    auto batch_strides = [](const std::vector<std::int64_t>& batch_shape, std::int64_t inner) {
        // inner = M*K (for A) or K*N (for B): the contiguous matrix
        // tile that follows the batch indexing.
        std::vector<std::int64_t> s(batch_shape.size(), 0);
        std::int64_t running = inner;
        for (std::int64_t i = static_cast<std::int64_t>(batch_shape.size()) - 1; i >= 0; --i) {
            const auto u = static_cast<std::size_t>(i);
            s[u] = running;
            running *= batch_shape[u];
        }
        return s;
    };

    const std::vector<std::int64_t> a_strides_full = batch_strides(a_batch, M * Ka);
    const std::vector<std::int64_t> b_strides_full = batch_strides(b_batch, Kb * N);

    // Right-align the per-input batch shapes/strides against the
    // broadcasted shape. Where a batch dim was broadcast from 1, set
    // its effective stride to 0.
    const std::size_t out_rank = bcast.size();
    std::vector<std::int64_t> a_eff_stride(out_rank, 0);
    std::vector<std::int64_t> b_eff_stride(out_rank, 0);
    for (std::size_t i = 0; i < a_batch.size(); ++i) {
        const std::size_t ai = a_batch.size() - 1 - i;
        const std::size_t oi = out_rank - 1 - i;
        a_eff_stride[oi] = a_batch[ai] == 1 ? 0 : a_strides_full[ai];
    }
    for (std::size_t i = 0; i < b_batch.size(); ++i) {
        const std::size_t bi = b_batch.size() - 1 - i;
        const std::size_t oi = out_rank - 1 - i;
        b_eff_stride[oi] = b_batch[bi] == 1 ? 0 : b_strides_full[bi];
    }

    // Iterate over the broadcasted batch shape, recording offsets.
    std::int64_t total_batch = 1;
    for (std::int64_t d : bcast) {
        total_batch *= d;
    }
    plan.a_offsets.resize(static_cast<std::size_t>(total_batch));
    plan.b_offsets.resize(static_cast<std::size_t>(total_batch));
    plan.c_offsets.resize(static_cast<std::size_t>(total_batch));

    if (total_batch > 0) {
        const std::int64_t c_step = M * N;
        for (std::int64_t step = 0; step < total_batch; ++step) {
            std::int64_t a_off = 0;
            std::int64_t b_off = 0;
            std::int64_t remainder = step;
            for (std::int64_t d = static_cast<std::int64_t>(out_rank) - 1; d >= 0; --d) {
                const auto u = static_cast<std::size_t>(d);
                const std::int64_t coord = remainder % bcast[u];
                remainder /= bcast[u];
                a_off += coord * a_eff_stride[u];
                b_off += coord * b_eff_stride[u];
            }
            plan.a_offsets[static_cast<std::size_t>(step)] = a_off;
            plan.b_offsets[static_cast<std::size_t>(step)] = b_off;
            plan.c_offsets[static_cast<std::size_t>(step)] = step * c_step;
        }
    }

    // User-visible output shape: bcast batch ++ {M, N}, with the
    // promoted axes squeezed back out for the 1-D input cases.
    plan.out_shape = bcast;
    if (!a_is_1d) {
        plan.out_shape.push_back(M);
    }
    if (!b_is_1d) {
        plan.out_shape.push_back(N);
    }
    // The 1-D × 1-D dot-product case: bcast is empty, M == N == 1, both
    // axes squeezed → 0-D scalar.

    return plan;
}

} // namespace ctorch::ops
