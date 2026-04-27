//===- src/ops/functors.h - Element-wise op functors -----------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tiny callable types injected into the templated CPU and CUDA kernels.
/// Defining each op once in a `__host__ __device__`-aware header makes it
/// impossible for the two backends' arithmetic to drift.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_OPS_FUNCTORS_H
#define CTORCH_OPS_FUNCTORS_H

#include <cmath>
#include <type_traits>

#if defined(__CUDACC__)
#define CTORCH_OP_FN __host__ __device__ inline
#else
#define CTORCH_OP_FN inline
#endif

namespace ctorch::ops {

// ---------- binary ----------

struct AddF {
    template <class T> CTORCH_OP_FN T operator()(T a, T b) const { return a + b; }
};
struct SubF {
    template <class T> CTORCH_OP_FN T operator()(T a, T b) const { return a - b; }
};
struct MulF {
    template <class T> CTORCH_OP_FN T operator()(T a, T b) const { return a * b; }
};
struct DivF {
    template <class T> CTORCH_OP_FN T operator()(T a, T b) const { return a / b; }
};

// ---------- unary, dtype-agnostic ----------

struct NegF {
    template <class T> CTORCH_OP_FN T operator()(T a) const { return -a; }
};

struct AbsF {
    template <class T> CTORCH_OP_FN T operator()(T a) const {
        if constexpr (std::is_unsigned_v<T>) {
            return a;
        } else {
            return a < T(0) ? -a : a;
        }
    }
};

struct ReluF {
    template <class T> CTORCH_OP_FN T operator()(T a) const { return a > T(0) ? a : T(0); }
};

// ---------- unary, transcendental (float-only) ----------

struct ExpF {
    CTORCH_OP_FN float operator()(float x) const { return ::expf(x); }
    CTORCH_OP_FN double operator()(double x) const { return ::exp(x); }
};

struct LogF {
    CTORCH_OP_FN float operator()(float x) const { return ::logf(x); }
    CTORCH_OP_FN double operator()(double x) const { return ::log(x); }
};

struct SqrtF {
    CTORCH_OP_FN float operator()(float x) const { return ::sqrtf(x); }
    CTORCH_OP_FN double operator()(double x) const { return ::sqrt(x); }
};

struct SigmoidF {
    CTORCH_OP_FN float operator()(float x) const {
        // 1 / (1 + exp(-x)). Stable for x >= 0; for very negative x we fall
        // back to exp(x) / (1 + exp(x)) to avoid blowing up the denominator.
        if (x >= 0.0f) {
            return 1.0f / (1.0f + ::expf(-x));
        }
        const float e = ::expf(x);
        return e / (1.0f + e);
    }
    CTORCH_OP_FN double operator()(double x) const {
        if (x >= 0.0) {
            return 1.0 / (1.0 + ::exp(-x));
        }
        const double e = ::exp(x);
        return e / (1.0 + e);
    }
};

struct TanhF {
    CTORCH_OP_FN float operator()(float x) const { return ::tanhf(x); }
    CTORCH_OP_FN double operator()(double x) const { return ::tanh(x); }
};

} // namespace ctorch::ops

#endif // CTORCH_OPS_FUNCTORS_H
