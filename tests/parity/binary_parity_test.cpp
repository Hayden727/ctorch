//===- tests/parity/binary_parity_test.cpp - .npy parity (binary) *- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loads each binary parity fixture under tests/parity/fixtures/ and
/// compares ctorch's CPU dispatcher output against the reference. Per
/// Issue 03 §N1, fp32 non-transcendentals tolerate 1e-5 relative error,
/// fp64 tolerates 1e-12. Integer ops require exact equality.
///
//===----------------------------------------------------------------------===//

#include "ctorch/dtype.h"
#include "ctorch/ops/elementwise.h"
#include "ctorch/tensor.h"

#include "load_npy.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifndef CTORCH_PARITY_FIXTURES_DIR
#error "CTORCH_PARITY_FIXTURES_DIR must be defined by the build system"
#endif

using ctorch::add;
using ctorch::div;
using ctorch::dtype;
using ctorch::mul;
using ctorch::sub;
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
    // alive for the body of the loop. Reading from `got.contiguous()`
    // directly would dangle if `got` was non-contiguous.
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
    ASSERT_EQ(got.dtype(), ref.dtype());
    const Tensor got_c = got.contiguous();
    const Tensor ref_c = ref.contiguous();
    const auto* g = data_of<T>(got_c);
    const auto* r = data_of<T>(ref_c);
    const std::int64_t n = ref.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(g[i], r[i]) << "i=" << i;
    }
}

// Auto-generated catalog mirrors scripts/gen_parity.py:BINARY_CASES so the
// loader can find each fixture by prefix without scanning the directory.
struct BinaryCase {
    const char* op;
    const char* prefix;
    dtype dt;
};

constexpr BinaryCase kBinary[] = {
    {"add", "add_float32_3x4_3x4", dtype::float32}, {"add", "add_float32_3x1_1x4", dtype::float32},
    {"add", "add_float64_5_5", dtype::float64},     {"add", "add_int32_4_4", dtype::int32},
    {"add", "add_int64_2x3_3", dtype::int64},       {"sub", "sub_float32_3x4_3x4", dtype::float32},
    {"sub", "sub_float64_2x2_2x2", dtype::float64}, {"sub", "sub_int32_5_5", dtype::int32},
    {"mul", "mul_float32_3x1_1x4", dtype::float32}, {"mul", "mul_float64_4_4", dtype::float64},
    {"mul", "mul_int32_3x3_3x3", dtype::int32},     {"div", "div_float32_4x4_4x4", dtype::float32},
    {"div", "div_float64_3_3", dtype::float64},
};

Tensor run_op(const std::string& op, const Tensor& a, const Tensor& b) {
    if (op == "add") {
        return add(a, b);
    }
    if (op == "sub") {
        return sub(a, b);
    }
    if (op == "mul") {
        return mul(a, b);
    }
    if (op == "div") {
        return div(a, b);
    }
    throw std::logic_error("unknown op: " + op);
}

double tol_for(dtype dt) {
    switch (dt) {
    case dtype::float32:
        return 1e-5;
    case dtype::float64:
        return 1e-12;
    default:
        return 0.0;
    }
}

class BinaryParity : public ::testing::TestWithParam<BinaryCase> {};

TEST_P(BinaryParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto a = load_npy(fixture_path(std::string(tc.prefix) + "_a.npy"));
    const auto b = load_npy(fixture_path(std::string(tc.prefix) + "_b.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto got = run_op(tc.op, a, b);
    switch (tc.dt) {
    case dtype::float32:
        expect_close<float>(got, ref, tol_for(tc.dt));
        break;
    case dtype::float64:
        expect_close<double>(got, ref, tol_for(tc.dt));
        break;
    case dtype::int32:
        expect_exact<std::int32_t>(got, ref);
        break;
    case dtype::int64:
        expect_exact<std::int64_t>(got, ref);
        break;
    case dtype::bool_:
    case dtype::bfloat16:
        FAIL() << "unsupported dtype in binary parity";
    }
}

INSTANTIATE_TEST_SUITE_P(All, BinaryParity, ::testing::ValuesIn(kBinary),
                         [](const ::testing::TestParamInfo<BinaryCase>& info) {
                             return std::string(info.param.prefix);
                         });

} // namespace
