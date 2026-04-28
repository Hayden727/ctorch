//===- tests/ops/reduction_dtype_test.cpp - dtype rules ------------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Black-box dtype-contract tests for reductions (issue 09 §F7):
///   * `sum`/`prod` of bool/int* promote to int64.
///   * `mean` requires a floating dtype.
///   * `max`/`min` preserve the input dtype.
///   * `argmax`/`argmin` always return int64.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/reduction.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::argmax;
using ctorch::argmin;
using ctorch::Device;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::max;
using ctorch::mean;
using ctorch::min;
using ctorch::prod;
using ctorch::sum;
using ctorch::Tensor;

namespace {

template <class T> Tensor scalar_filled(dtype dt, T value) {
    Tensor t({3}, dt, Device::cpu());
    auto* p = static_cast<T*>(t.storage().data());
    p[0] = p[1] = p[2] = value;
    return t;
}

} // namespace

TEST(ReductionDTypeRules, SumPromotesBoolToInt64) {
    Tensor t({3}, dtype::bool_, Device::cpu());
    auto* p = static_cast<unsigned char*>(t.storage().data());
    p[0] = 1;
    p[1] = 0;
    p[2] = 1;
    const auto y = sum(t);
    EXPECT_EQ(y.dtype(), dtype::int64);
    EXPECT_EQ(*static_cast<const std::int64_t*>(y.storage().data()), 2);
}

TEST(ReductionDTypeRules, SumPromotesInt32ToInt64) {
    auto t = scalar_filled<std::int32_t>(dtype::int32, 7);
    EXPECT_EQ(sum(t).dtype(), dtype::int64);
}

TEST(ReductionDTypeRules, SumPreservesFloatDtype) {
    auto f32 = scalar_filled<float>(dtype::float32, 1.0f);
    auto f64 = scalar_filled<double>(dtype::float64, 1.0);
    EXPECT_EQ(sum(f32).dtype(), dtype::float32);
    EXPECT_EQ(sum(f64).dtype(), dtype::float64);
}

TEST(ReductionDTypeRules, ProdPromotesIntegralToInt64) {
    auto t = scalar_filled<std::int32_t>(dtype::int32, 2);
    EXPECT_EQ(prod(t).dtype(), dtype::int64);
}

TEST(ReductionDTypeRules, MeanRequiresFloatingDtype) {
    auto i32 = scalar_filled<std::int32_t>(dtype::int32, 1);
    auto i64 = scalar_filled<std::int64_t>(dtype::int64, 1);
    Tensor b({3}, dtype::bool_, Device::cpu());
    EXPECT_THROW(mean(i32), DTypeError);
    EXPECT_THROW(mean(i64), DTypeError);
    EXPECT_THROW(mean(b), DTypeError);
}

TEST(ReductionDTypeRules, MaxMinPreserveDtype) {
    auto f32 = scalar_filled<float>(dtype::float32, 1.0f);
    auto i32 = scalar_filled<std::int32_t>(dtype::int32, 1);
    EXPECT_EQ(max(f32).dtype(), dtype::float32);
    EXPECT_EQ(min(f32).dtype(), dtype::float32);
    EXPECT_EQ(max(i32).dtype(), dtype::int32);
    EXPECT_EQ(min(i32).dtype(), dtype::int32);
}

TEST(ReductionDTypeRules, MaxSingleAxisValuesIndicesShape) {
    auto f32 = scalar_filled<float>(dtype::float32, 1.0f);
    auto vi = max(f32, /*dim=*/0);
    EXPECT_EQ(vi.values.dtype(), dtype::float32);
    EXPECT_EQ(vi.indices.dtype(), dtype::int64);
}

TEST(ReductionDTypeRules, ArgmaxArgminAlwaysInt64) {
    auto f32 = scalar_filled<float>(dtype::float32, 1.0f);
    auto i32 = scalar_filled<std::int32_t>(dtype::int32, 1);
    EXPECT_EQ(argmax(f32, 0).dtype(), dtype::int64);
    EXPECT_EQ(argmin(i32, 0).dtype(), dtype::int64);
}

TEST(ReductionDTypeRules, AllOpsRejectBfloat16) {
    Tensor bf({3}, dtype::bfloat16, Device::cpu());
    EXPECT_THROW(sum(bf), DTypeError);
    EXPECT_THROW(prod(bf), DTypeError);
    EXPECT_THROW(mean(bf), DTypeError);
    EXPECT_THROW(max(bf), DTypeError);
    EXPECT_THROW(min(bf), DTypeError);
    EXPECT_THROW(argmax(bf, 0), DTypeError);
    EXPECT_THROW(argmin(bf, 0), DTypeError);
}
