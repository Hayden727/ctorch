//===- tests/parity/matmul_parity_test.cpp - .npy parity (matmul) ---------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loads each matmul parity fixture under tests/parity/fixtures/ and
/// compares ctorch's CPU dispatcher output against the NumPy reference.
/// Per Issue #11 §N1 the tolerance is `1e-4` rel for fp32 (BLAS uses
/// fused multiply-add and drifts measurably from the naïve reference)
/// and `1e-12` rel for fp64.
///
//===----------------------------------------------------------------------===//

#include "ctorch/dtype.h"
#include "ctorch/ops/linalg.h"
#include "ctorch/tensor.h"

#include "load_npy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#ifndef CTORCH_PARITY_FIXTURES_DIR
#error "CTORCH_PARITY_FIXTURES_DIR must be defined by the build system"
#endif

using ctorch::dtype;
using ctorch::matmul;
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

void compare(const Tensor& got, const Tensor& ref) {
    switch (ref.dtype()) {
    case dtype::float32:
        expect_close<float>(got, ref, 1e-4);
        break;
    case dtype::float64:
        expect_close<double>(got, ref, 1e-12);
        break;
    default:
        FAIL() << "unsupported dtype in matmul parity";
    }
}

struct MatmulCase {
    const char* prefix;
};

constexpr MatmulCase kCases[] = {
    {"matmul_float32_3_x_3"},       {"matmul_float32_4_x_4x3"},
    {"matmul_float32_3x4_x_4"},     {"matmul_float32_5x7_x_7x11"},
    {"matmul_float64_3x4_x_4x5"},   {"matmul_float32_2x3x4_x_2x4x5"},
    {"matmul_float32_2x3x4_x_4x5"}, {"matmul_float32_1x3x4_x_5x4x6"},
};

class MatmulParity : public ::testing::TestWithParam<MatmulCase> {};

TEST_P(MatmulParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto a = load_npy(fixture_path(std::string(tc.prefix) + "_a.npy"));
    const auto b = load_npy(fixture_path(std::string(tc.prefix) + "_b.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto got = matmul(a, b);
    compare(got, ref);
}

INSTANTIATE_TEST_SUITE_P(All, MatmulParity, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<MatmulCase>& info) {
                             return std::string(info.param.prefix);
                         });

} // namespace
