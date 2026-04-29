//===- tests/parity/indexing_parity_test.cpp - .npy parity (indexing) -----===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loads each indexing parity fixture under tests/parity/fixtures/ and
/// compares ctorch's CPU dispatcher output for `index_select` against
/// the NumPy reference (`np.take` with `axis=dim`). Integer-exact
/// because gather is a pure copy — no rounding involved.
///
//===----------------------------------------------------------------------===//

#include "ctorch/dtype.h"
#include "ctorch/ops/indexing.h"
#include "ctorch/tensor.h"

#include "load_npy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#ifndef CTORCH_PARITY_FIXTURES_DIR
#error "CTORCH_PARITY_FIXTURES_DIR must be defined by the build system"
#endif

using ctorch::dtype;
using ctorch::index_select;
using ctorch::Tensor;
using ctorch::parity::load_npy;

namespace {

std::string fixture_path(const std::string& name) {
    return std::string(CTORCH_PARITY_FIXTURES_DIR) + "/" + name;
}

template <class T> const T* data_of(const Tensor& t) {
    return static_cast<const T*>(t.storage().data()) + t.offset();
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

void compare_exact(const Tensor& got, const Tensor& ref) {
    switch (ref.dtype()) {
    case dtype::float32:
        expect_exact<float>(got, ref);
        break;
    case dtype::float64:
        expect_exact<double>(got, ref);
        break;
    case dtype::int32:
        expect_exact<std::int32_t>(got, ref);
        break;
    case dtype::int64:
        expect_exact<std::int64_t>(got, ref);
        break;
    default:
        FAIL() << "unsupported dtype in indexing parity";
    }
}

struct IndexSelectCase {
    const char* prefix;
    int dim;
};

constexpr IndexSelectCase kCases[] = {
    {"index_select_float32_4x3_dim0_int64_n4", 0},
    {"index_select_float32_3x4_dim1_int32_n3", 1},
    {"index_select_float64_5_dim0_int64_n3", 0},
    {"index_select_int32_3x4_dim0_int64_n3", 0},
    {"index_select_int64_2x3x4_dim2_int32_n2", 2},
    {"index_select_float32_2x3x4_dim1_int64_n3", 1},
};

class IndexSelectParity : public ::testing::TestWithParam<IndexSelectCase> {};

TEST_P(IndexSelectParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto x = load_npy(fixture_path(std::string(tc.prefix) + "_in.npy"));
    const auto idx = load_npy(fixture_path(std::string(tc.prefix) + "_idx.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto got = index_select(x, tc.dim, idx);
    compare_exact(got, ref);
}

INSTANTIATE_TEST_SUITE_P(All, IndexSelectParity, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<IndexSelectCase>& info) {
                             return std::string(info.param.prefix);
                         });

} // namespace
