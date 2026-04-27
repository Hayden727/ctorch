//===- tests/parity/unary_parity_test.cpp - .npy parity (unary) -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ctorch/dtype.h"
#include "ctorch/ops/elementwise.h"
#include "ctorch/tensor.h"

#include "load_npy.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

#ifndef CTORCH_PARITY_FIXTURES_DIR
#error "CTORCH_PARITY_FIXTURES_DIR must be defined by the build system"
#endif

using ctorch::abs;
using ctorch::dtype;
using ctorch::exp;
using ctorch::log;
using ctorch::neg;
using ctorch::relu;
using ctorch::sigmoid;
using ctorch::sqrt;
using ctorch::tanh;
using ctorch::Tensor;
using ctorch::parity::load_npy;

namespace {

std::string fixture_path(const std::string& name) {
    return std::string(CTORCH_PARITY_FIXTURES_DIR) + "/" + name;
}

template <class T> const T* data_of(const Tensor& t) {
    return static_cast<const T*>(t.storage().data()) + t.offset();
}

template <class T> void expect_close(const Tensor& got, const Tensor& ref, double tol) {
    ASSERT_EQ(got.shape(), ref.shape());
    ASSERT_EQ(got.dtype(), ref.dtype());
    // Bind the contiguous results to named tensors so their Storage stays
    // alive for the body of the loop.
    const Tensor got_c = got.contiguous();
    const Tensor ref_c = ref.contiguous();
    const auto* g = data_of<T>(got_c);
    const auto* r = data_of<T>(ref_c);
    const std::int64_t n = ref.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        const double diff = std::abs(static_cast<double>(g[i]) - static_cast<double>(r[i]));
        const double scale = std::max(1.0, std::abs(static_cast<double>(r[i])));
        EXPECT_LE(diff, tol * scale) << "i=" << i << " got=" << g[i] << " ref=" << r[i];
    }
}

template <class T> void expect_exact(const Tensor& got, const Tensor& ref) {
    ASSERT_EQ(got.shape(), ref.shape());
    const Tensor got_c = got.contiguous();
    const Tensor ref_c = ref.contiguous();
    const auto* g = data_of<T>(got_c);
    const auto* r = data_of<T>(ref_c);
    const std::int64_t n = ref.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(g[i], r[i]) << "i=" << i;
    }
}

struct UnaryCase {
    const char* op;
    const char* prefix;
    dtype dt;
    bool transcendental;
};

constexpr UnaryCase kUnary[] = {
    {"neg", "neg_float32_3x4", dtype::float32, false},
    {"neg", "neg_int32_5", dtype::int32, false},
    {"abs", "abs_float32_3x4", dtype::float32, false},
    {"abs", "abs_int64_5", dtype::int64, false},
    {"relu", "relu_float32_8", dtype::float32, false},
    {"exp", "exp_float32_4", dtype::float32, true},
    {"exp", "exp_float64_4", dtype::float64, true},
    {"log", "log_float32_4", dtype::float32, true},
    {"log", "log_float64_4", dtype::float64, true},
    {"sqrt", "sqrt_float32_4", dtype::float32, true},
    {"sqrt", "sqrt_float64_4", dtype::float64, true},
    {"sigmoid", "sigmoid_float32_8", dtype::float32, true},
    {"sigmoid", "sigmoid_float64_8", dtype::float64, true},
    {"tanh", "tanh_float32_8", dtype::float32, true},
    {"tanh", "tanh_float64_8", dtype::float64, true},
};

Tensor run_op(const std::string& op, const Tensor& x) {
    if (op == "neg") {
        return neg(x);
    }
    if (op == "abs") {
        return abs(x);
    }
    if (op == "relu") {
        return relu(x);
    }
    if (op == "exp") {
        return exp(x);
    }
    if (op == "log") {
        return log(x);
    }
    if (op == "sqrt") {
        return sqrt(x);
    }
    if (op == "sigmoid") {
        return sigmoid(x);
    }
    if (op == "tanh") {
        return tanh(x);
    }
    throw std::logic_error("unknown op: " + op);
}

double tol_for(dtype dt, bool transcendental) {
    switch (dt) {
    case dtype::float32:
        return transcendental ? 1e-4 : 1e-5;
    case dtype::float64:
        return 1e-12;
    default:
        return 0.0;
    }
}

class UnaryParity : public ::testing::TestWithParam<UnaryCase> {};

TEST_P(UnaryParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto in = load_npy(fixture_path(std::string(tc.prefix) + "_in.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto got = run_op(tc.op, in);
    switch (tc.dt) {
    case dtype::float32:
        expect_close<float>(got, ref, tol_for(tc.dt, tc.transcendental));
        break;
    case dtype::float64:
        expect_close<double>(got, ref, tol_for(tc.dt, tc.transcendental));
        break;
    case dtype::int32:
        expect_exact<std::int32_t>(got, ref);
        break;
    case dtype::int64:
        expect_exact<std::int64_t>(got, ref);
        break;
    case dtype::bool_:
    case dtype::bfloat16:
        FAIL() << "unsupported dtype in unary parity";
    }
}

INSTANTIATE_TEST_SUITE_P(All, UnaryParity, ::testing::ValuesIn(kUnary),
                         [](const ::testing::TestParamInfo<UnaryCase>& info) {
                             return std::string(info.param.prefix);
                         });

} // namespace
