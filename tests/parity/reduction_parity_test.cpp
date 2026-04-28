//===- tests/parity/reduction_parity_test.cpp - .npy parity (reductions) ===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loads each reduction parity fixture under tests/parity/fixtures/ and
/// compares ctorch's CPU dispatcher output against the reference. Per
/// issue 09 §N1, fp32 tolerates 1e-5 relative error, fp64 tolerates
/// 1e-12, integer reductions and index reductions require exact
/// equality.
///
//===----------------------------------------------------------------------===//

#include "ctorch/dtype.h"
#include "ctorch/ops/reduction.h"
#include "ctorch/tensor.h"

#include "load_npy.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifndef CTORCH_PARITY_FIXTURES_DIR
#error "CTORCH_PARITY_FIXTURES_DIR must be defined by the build system"
#endif

using ctorch::argmax;
using ctorch::argmin;
using ctorch::dtype;
using ctorch::max;
using ctorch::mean;
using ctorch::min;
using ctorch::prod;
using ctorch::sum;
using ctorch::Tensor;
using ctorch::ValuesIndices;
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

void compare(const Tensor& got, const Tensor& ref) {
    switch (ref.dtype()) {
    case dtype::float32:
        expect_close<float>(got, ref, tol_for(dtype::float32));
        break;
    case dtype::float64:
        expect_close<double>(got, ref, tol_for(dtype::float64));
        break;
    case dtype::int32:
        expect_exact<std::int32_t>(got, ref);
        break;
    case dtype::int64:
        expect_exact<std::int64_t>(got, ref);
        break;
    default:
        FAIL() << "unsupported dtype in reduction parity";
    }
}

// ---------- sum-like fixtures (sum/mean/prod) ----------

struct SumLikeCase {
    const char* op;
    const char* prefix;
    const char* dims; // "all" or comma-separated, e.g. "1" / "0,2"
    bool keepdim;
};

constexpr SumLikeCase kSumLike[] = {
    {"sum", "sum_float32_3x4_dimsAll_kd0", "all", false},
    {"sum", "sum_float32_3x4_dims1_kd0", "1", false},
    {"sum", "sum_float32_3x4_dims1_kd1", "1", true},
    {"sum", "sum_float32_2x3x4_dims0_2_kd0", "0,2", false},
    {"sum", "sum_float64_8_dimsAll_kd0", "all", false},
    {"sum", "sum_int32_4x4_dimsAll_kd0", "all", false},
    {"sum", "sum_int32_3x4_dims0_kd0", "0", false},
    {"mean", "mean_float32_3x4_dims1_kd0", "1", false},
    {"mean", "mean_float64_5_dimsAll_kd0", "all", false},
    {"mean", "mean_float32_2x3x4_dims0_2_kd1", "0,2", true},
    {"prod", "prod_float32_3_dimsAll_kd0", "all", false},
    {"prod", "prod_int32_4_dimsAll_kd0", "all", false},
    {"prod", "prod_float32_2x3_dims1_kd0", "1", false},
};

std::vector<std::int64_t> parse_dims(const char* s) {
    std::vector<std::int64_t> out;
    if (std::string(s) == "all") {
        return out;
    }
    std::string buf;
    for (const char* c = s;; ++c) {
        if (*c == ',' || *c == '\0') {
            if (!buf.empty()) {
                out.push_back(std::stoll(buf));
                buf.clear();
            }
            if (*c == '\0') {
                break;
            }
        } else {
            buf += *c;
        }
    }
    return out;
}

class SumLikeParity : public ::testing::TestWithParam<SumLikeCase> {};

TEST_P(SumLikeParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto x = load_npy(fixture_path(std::string(tc.prefix) + "_in.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto dims = parse_dims(tc.dims);
    Tensor got;
    if (std::string(tc.op) == "sum") {
        got = dims.empty() ? sum(x) : sum(x, dims, tc.keepdim);
    } else if (std::string(tc.op) == "mean") {
        got = dims.empty() ? mean(x) : mean(x, dims, tc.keepdim);
    } else if (std::string(tc.op) == "prod") {
        got = dims.empty() ? prod(x) : prod(x, dims, tc.keepdim);
    } else {
        FAIL() << "unknown sum-like op";
    }
    compare(got, ref);
}

INSTANTIATE_TEST_SUITE_P(All, SumLikeParity, ::testing::ValuesIn(kSumLike),
                         [](const ::testing::TestParamInfo<SumLikeCase>& info) {
                             return std::string(info.param.prefix);
                         });

// ---------- max/min value fixtures (multi-axis or whole-tensor) ----------

struct MaxMinValueCase {
    const char* op; // "max" or "min"
    const char* prefix;
    const char* dims;
    bool keepdim;
};

constexpr MaxMinValueCase kMaxMinValue[] = {
    {"max", "maxval_float32_4_dimsAll_kd0", "all", false},
    {"max", "maxval_float32_3x4_dims1_kd0", "1", false},
    {"max", "maxval_int32_3x4_dims0_kd1", "0", true},
    {"min", "minval_float32_3x4_dims0_kd1", "0", true},
    {"min", "minval_int64_3x4_dimsAll_kd0", "all", false},
};

class MaxMinValueParity : public ::testing::TestWithParam<MaxMinValueCase> {};

TEST_P(MaxMinValueParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto x = load_npy(fixture_path(std::string(tc.prefix) + "_in.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto dims = parse_dims(tc.dims);
    Tensor got;
    if (std::string(tc.op) == "max") {
        got = dims.empty() ? max(x) : max(x, dims, tc.keepdim);
    } else {
        got = dims.empty() ? min(x) : min(x, dims, tc.keepdim);
    }
    compare(got, ref);
}

INSTANTIATE_TEST_SUITE_P(All, MaxMinValueParity, ::testing::ValuesIn(kMaxMinValue),
                         [](const ::testing::TestParamInfo<MaxMinValueCase>& info) {
                             return std::string(info.param.prefix);
                         });

// ---------- max/min single-axis with indices ----------

struct MaxMinIdxCase {
    const char* op;
    const char* prefix;
    std::int64_t axis;
    bool keepdim;
};

constexpr MaxMinIdxCase kMaxMinIdx[] = {
    {"max", "maxidx_float32_3x4_dim1_kd0", 1, false},
    {"min", "minidx_int32_3x2_dim0_kd0", 0, false},
    {"max", "maxidx_float32_2x3x4_dimn1_kd1", -1, true},
};

class MaxMinIdxParity : public ::testing::TestWithParam<MaxMinIdxCase> {};

TEST_P(MaxMinIdxParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto x = load_npy(fixture_path(std::string(tc.prefix) + "_in.npy"));
    const auto ref_val = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto ref_idx = load_npy(fixture_path(std::string(tc.prefix) + "_ref_idx.npy"));
    ValuesIndices vi;
    if (std::string(tc.op) == "max") {
        vi = max(x, tc.axis, tc.keepdim);
    } else {
        vi = min(x, tc.axis, tc.keepdim);
    }
    compare(vi.values, ref_val);
    expect_exact<std::int64_t>(vi.indices, ref_idx);
}

INSTANTIATE_TEST_SUITE_P(All, MaxMinIdxParity, ::testing::ValuesIn(kMaxMinIdx),
                         [](const ::testing::TestParamInfo<MaxMinIdxCase>& info) {
                             return std::string(info.param.prefix);
                         });

// ---------- argmax / argmin ----------

struct ArgCase {
    const char* op;
    const char* prefix;
    std::int64_t axis;
    bool keepdim;
};

constexpr ArgCase kArg[] = {
    {"argmax", "argmax_float32_3x4_dim1_kd0", 1, false},
    {"argmin", "argmin_int64_5_dim0_kd0", 0, false},
    {"argmax", "argmax_float32_5_dim0_kd0_tied", 0, false},
};

class ArgParity : public ::testing::TestWithParam<ArgCase> {};

TEST_P(ArgParity, MatchesReference) {
    const auto& tc = GetParam();
    const auto x = load_npy(fixture_path(std::string(tc.prefix) + "_in.npy"));
    const auto ref = load_npy(fixture_path(std::string(tc.prefix) + "_ref.npy"));
    const auto got = std::string(tc.op) == "argmax" ? argmax(x, tc.axis, tc.keepdim)
                                                    : argmin(x, tc.axis, tc.keepdim);
    expect_exact<std::int64_t>(got, ref);
}

INSTANTIATE_TEST_SUITE_P(All, ArgParity, ::testing::ValuesIn(kArg),
                         [](const ::testing::TestParamInfo<ArgCase>& info) {
                             return std::string(info.param.prefix);
                         });

} // namespace
