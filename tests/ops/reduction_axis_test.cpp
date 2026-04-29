//===- tests/ops/reduction_axis_test.cpp - axis canonicalisation ---------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// White-box tests for `ctorch::ops::canonicalise` and the output-shape
/// helpers. Reaches into the private header under src/ops/.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/tensor.h"

#include "ops/reduction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::ShapeError;
using ctorch::Tensor;
using ctorch::ops::canonicalise;
using ctorch::ops::canonicalise_single;
using ctorch::ops::reduce_sum_prod_dtype;
using ctorch::ops::reduced_shape;
using ctorch::ops::reduced_shape_single;
using ctorch::ops::require_float_for_mean;

namespace {

Tensor t_of(std::vector<std::int64_t> shape, dtype dt = dtype::float32) {
    return Tensor(std::move(shape), dt, Device::cpu());
}

} // namespace

TEST(ReductionAxis, EmptyDimsCollapsesEveryAxis) {
    auto x = t_of({2, 3, 4});
    const auto ax = canonicalise(x, {});
    EXPECT_EQ(ax.rank, 3);
    EXPECT_TRUE(ax.reduce[0]);
    EXPECT_TRUE(ax.reduce[1]);
    EXPECT_TRUE(ax.reduce[2]);
    EXPECT_EQ(ax.reduced_numel, 24);
    EXPECT_EQ(ax.kept_numel, 1);
}

TEST(ReductionAxis, NegativeAxisIsNormalised) {
    auto x = t_of({2, 3, 4});
    const auto ax = canonicalise(x, {-1});
    EXPECT_TRUE(ax.reduce[2]);
    EXPECT_FALSE(ax.reduce[0]);
    EXPECT_FALSE(ax.reduce[1]);
    EXPECT_EQ(ax.reduced_numel, 4);
    EXPECT_EQ(ax.kept_numel, 6);
}

TEST(ReductionAxis, MultipleAxesAccumulateNumels) {
    auto x = t_of({2, 3, 4});
    const auto ax = canonicalise(x, {0, 2});
    EXPECT_TRUE(ax.reduce[0]);
    EXPECT_FALSE(ax.reduce[1]);
    EXPECT_TRUE(ax.reduce[2]);
    EXPECT_EQ(ax.reduced_numel, 8);
    EXPECT_EQ(ax.kept_numel, 3);
}

TEST(ReductionAxis, DuplicateAxisThrowsShapeError) {
    auto x = t_of({2, 3, 4});
    EXPECT_THROW(canonicalise(x, {1, 1}), ShapeError);
    // Negative + positive form of same axis also count as a duplicate.
    EXPECT_THROW(canonicalise(x, {1, -2}), ShapeError);
}

TEST(ReductionAxis, OutOfRangeAxisThrowsShapeError) {
    auto x = t_of({2, 3, 4});
    EXPECT_THROW(canonicalise(x, {3}), ShapeError);
    EXPECT_THROW(canonicalise(x, {-4}), ShapeError);
}

TEST(ReductionAxis, ZeroDimTensorEmptyDimsIsNoop) {
    auto x = t_of({});
    const auto ax = canonicalise(x, {});
    EXPECT_EQ(ax.rank, 0);
    EXPECT_EQ(ax.reduced_numel, 1);
    EXPECT_EQ(ax.kept_numel, 1);
}

TEST(ReductionAxis, ReducedShapeKeepDimReplacesWithOne) {
    auto x = t_of({2, 3, 4});
    const auto ax = canonicalise(x, {1});
    EXPECT_EQ(reduced_shape(x, ax, /*keepdim=*/true), (std::vector<std::int64_t>{2, 1, 4}));
    EXPECT_EQ(reduced_shape(x, ax, /*keepdim=*/false), (std::vector<std::int64_t>{2, 4}));
}

TEST(ReductionAxis, ReducedShapeWholeTensorIsZeroDOrAllOnes) {
    auto x = t_of({2, 3, 4});
    const auto ax = canonicalise(x, {});
    EXPECT_TRUE(reduced_shape(x, ax, /*keepdim=*/false).empty());
    EXPECT_EQ(reduced_shape(x, ax, /*keepdim=*/true), (std::vector<std::int64_t>{1, 1, 1}));
}

TEST(ReductionAxis, SingleAxisCanonicalisationRejectsZeroDim) {
    auto x = t_of({});
    EXPECT_THROW(canonicalise_single(x, 0), ShapeError);
}

TEST(ReductionAxis, SingleAxisCanonicalisationNormalises) {
    auto x = t_of({2, 3, 4});
    EXPECT_EQ(canonicalise_single(x, -1), 2);
    EXPECT_EQ(canonicalise_single(x, 0), 0);
    EXPECT_THROW(canonicalise_single(x, 3), ShapeError);
}

TEST(ReductionAxis, SingleAxisReducedShape) {
    auto x = t_of({2, 3, 4});
    EXPECT_EQ(reduced_shape_single(x, 1, /*keepdim=*/true), (std::vector<std::int64_t>{2, 1, 4}));
    EXPECT_EQ(reduced_shape_single(x, 1, /*keepdim=*/false), (std::vector<std::int64_t>{2, 4}));
}

TEST(ReductionDtype, SumProdPromotesIntegralToInt64) {
    EXPECT_EQ(reduce_sum_prod_dtype(dtype::bool_), dtype::int64);
    EXPECT_EQ(reduce_sum_prod_dtype(dtype::int32), dtype::int64);
    EXPECT_EQ(reduce_sum_prod_dtype(dtype::int64), dtype::int64);
    EXPECT_EQ(reduce_sum_prod_dtype(dtype::float32), dtype::float32);
    EXPECT_EQ(reduce_sum_prod_dtype(dtype::float64), dtype::float64);
    EXPECT_THROW(reduce_sum_prod_dtype(dtype::bfloat16), DTypeError);
}

TEST(ReductionDtype, MeanRejectsIntegralInputs) {
    EXPECT_THROW(require_float_for_mean(dtype::int32, "mean"), DTypeError);
    EXPECT_THROW(require_float_for_mean(dtype::bool_, "mean"), DTypeError);
    EXPECT_NO_THROW(require_float_for_mean(dtype::float32, "mean"));
    EXPECT_NO_THROW(require_float_for_mean(dtype::float64, "mean"));
}
