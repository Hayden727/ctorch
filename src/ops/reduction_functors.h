//===- src/ops/reduction_functors.h - Reduction op functors ----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// `__host__ __device__`-aware reduction functors injected into the
/// templated CPU and CUDA kernels. Each functor exposes:
///   * `static constexpr Acc identity<Acc>()` — accumulator seed.
///   * `void apply(Acc& acc, T v)` — fold a single input.
///
/// `Accumulator<Op, T>` picks the widening accumulator dtype per
/// (op, input_t) tuple to match the issue 09 §F7 dtype contract and
/// mitigate the §7 risk-table item on CPU/CUDA fp drift (we widen
/// fp32 sums to double on both sides).
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_REDUCTION_FUNCTORS_H
#define CTORCH_OPS_REDUCTION_FUNCTORS_H

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define CTORCH_REDUCE_FN __host__ __device__ inline
#else
#define CTORCH_REDUCE_FN inline
#endif

namespace ctorch::ops {

struct SumF {
    template <class Acc> static CTORCH_REDUCE_FN Acc identity() { return Acc{0}; }
    template <class Acc, class T> static CTORCH_REDUCE_FN void apply(Acc& acc, T v) {
        acc += static_cast<Acc>(v);
    }
};

struct ProdF {
    template <class Acc> static CTORCH_REDUCE_FN Acc identity() { return Acc{1}; }
    template <class Acc, class T> static CTORCH_REDUCE_FN void apply(Acc& acc, T v) {
        acc *= static_cast<Acc>(v);
    }
};

/// NaN-propagating max / min: if any input is NaN the result is NaN
/// (PyTorch convention). Once `acc` is NaN we keep it; if a fresh `v`
/// is NaN we adopt it. Float identity is -inf / +inf so the very first
/// finite element wins; integer identity is the type's lowest / max.
struct MaxF {
    template <class Acc> static CTORCH_REDUCE_FN Acc identity() {
        if constexpr (std::is_floating_point_v<Acc>) {
            return -std::numeric_limits<Acc>::infinity();
        } else {
            return std::numeric_limits<Acc>::lowest();
        }
    }
    template <class Acc, class T> static CTORCH_REDUCE_FN void apply(Acc& acc, T v) {
        if (should_replace<Acc>(acc, static_cast<Acc>(v))) {
            acc = static_cast<Acc>(v);
        }
    }
    /// True iff a fresh value `v` should overwrite the running best
    /// `cur`. Strict comparison so that ties keep the **first**
    /// occurrence (matches PyTorch's argmax tie-breaking). NaN
    /// propagation: once `cur` is NaN nothing replaces it; an incoming
    /// NaN replaces a non-NaN `cur` (so the first NaN's index wins).
    template <class Acc> static CTORCH_REDUCE_FN bool should_replace(Acc cur, Acc v) {
        if constexpr (std::is_floating_point_v<Acc>) {
            if (cur != cur) {
                return false;
            }
            if (v != v) {
                return true;
            }
        }
        return v > cur;
    }
};

struct MinF {
    template <class Acc> static CTORCH_REDUCE_FN Acc identity() {
        if constexpr (std::is_floating_point_v<Acc>) {
            return std::numeric_limits<Acc>::infinity();
        } else {
            return std::numeric_limits<Acc>::max();
        }
    }
    template <class Acc, class T> static CTORCH_REDUCE_FN void apply(Acc& acc, T v) {
        if (should_replace<Acc>(acc, static_cast<Acc>(v))) {
            acc = static_cast<Acc>(v);
        }
    }
    template <class Acc> static CTORCH_REDUCE_FN bool should_replace(Acc cur, Acc v) {
        if constexpr (std::is_floating_point_v<Acc>) {
            if (cur != cur) {
                return false;
            }
            if (v != v) {
                return true;
            }
        }
        return v < cur;
    }
};

} // namespace ctorch::ops

#endif // CTORCH_OPS_REDUCTION_FUNCTORS_H
