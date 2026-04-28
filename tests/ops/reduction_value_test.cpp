//===- tests/ops/reduction_value_test.cpp - functional reductions --------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functional tests for the public reduction free functions on CPU
/// tensors. Covers whole-tensor and axis forms with hand-computed
/// expected values; parity against PyTorch references is tested
/// separately under `tests/parity/reduction_parity_test.cpp`.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/reduction.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
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
using ctorch::ShapeError;
using ctorch::sum;
using ctorch::Tensor;
using ctorch::ValuesIndices;

namespace {

template <class T>
Tensor make_filled(std::vector<std::int64_t> shape, dtype dt, std::initializer_list<T> values) {
    Tensor t(std::move(shape), dt, Device::cpu());
    auto* p = static_cast<T*>(t.storage().data());
    std::int64_t i = 0;
    for (auto v : values) {
        p[i++] = v;
    }
    return t;
}

template <class T> std::vector<T> read_all(const Tensor& t) {
    const auto* p = static_cast<const T*>(t.storage().data()) + t.offset();
    return std::vector<T>(p, p + t.numel());
}

} // namespace

TEST(ReductionSum, WholeTensorFloat32) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    auto y = sum(x);
    EXPECT_EQ(y.dtype(), dtype::float32);
    EXPECT_TRUE(y.shape().empty());
    EXPECT_FLOAT_EQ(read_all<float>(y)[0], 21.0f);
}

TEST(ReductionSum, WholeTensorIntPromotesToInt64) {
    auto x = make_filled<std::int32_t>({4}, dtype::int32, {1, 2, 3, 4});
    auto y = sum(x);
    EXPECT_EQ(y.dtype(), dtype::int64);
    EXPECT_EQ(read_all<std::int64_t>(y)[0], 10);
}

TEST(ReductionSum, AxisFormFloat32) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    auto y = sum(x, {1});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(read_all<float>(y), (std::vector<float>{6.0f, 15.0f}));
}

TEST(ReductionSum, AxisFormKeepdim) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    auto y = sum(x, {1}, /*keepdim=*/true);
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{2, 1}));
    EXPECT_EQ(read_all<float>(y), (std::vector<float>{6.0f, 15.0f}));
}

TEST(ReductionSum, MultiAxis) {
    auto x = make_filled<float>({2, 2, 2}, dtype::float32, {1, 2, 3, 4, 5, 6, 7, 8});
    auto y = sum(x, {0, 2});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{2}));
    // i=0: x[:,0,:] = [[1,2],[5,6]] = 14; i=1: x[:,1,:] = [[3,4],[7,8]] = 22.
    EXPECT_EQ(read_all<float>(y), (std::vector<float>{14.0f, 22.0f}));
}

TEST(ReductionSum, NegativeAxis) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    auto y = sum(x, {-1});
    EXPECT_EQ(read_all<float>(y), (std::vector<float>{6.0f, 15.0f}));
}

TEST(ReductionMean, WholeTensorFloat32) {
    auto x = make_filled<float>({4}, dtype::float32, {1, 2, 3, 4});
    auto y = mean(x);
    EXPECT_FLOAT_EQ(read_all<float>(y)[0], 2.5f);
}

TEST(ReductionMean, AxisFormFloat64) {
    auto x = make_filled<double>({2, 3}, dtype::float64, {1, 2, 3, 4, 5, 6});
    auto y = mean(x, {0});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{3}));
    EXPECT_DOUBLE_EQ(read_all<double>(y)[0], 2.5);
    EXPECT_DOUBLE_EQ(read_all<double>(y)[1], 3.5);
    EXPECT_DOUBLE_EQ(read_all<double>(y)[2], 4.5);
}

TEST(ReductionMean, RejectsIntegralInput) {
    auto x = make_filled<std::int32_t>({4}, dtype::int32, {1, 2, 3, 4});
    EXPECT_THROW(mean(x), DTypeError);
}

TEST(ReductionProd, WholeTensorFloat32) {
    auto x = make_filled<float>({4}, dtype::float32, {1, 2, 3, 4});
    auto y = prod(x);
    EXPECT_FLOAT_EQ(read_all<float>(y)[0], 24.0f);
}

TEST(ReductionProd, AxisFormInt32PromotesToInt64) {
    auto x = make_filled<std::int32_t>({2, 3}, dtype::int32, {1, 2, 3, 4, 5, 6});
    auto y = prod(x, {1});
    EXPECT_EQ(y.dtype(), dtype::int64);
    EXPECT_EQ(read_all<std::int64_t>(y), (std::vector<std::int64_t>{6, 120}));
}

TEST(ReductionSum, EmptySliceProducesZero) {
    Tensor x({0, 3}, dtype::float32, Device::cpu());
    auto y = sum(x, {0});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{3}));
    for (float v : read_all<float>(y)) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(ReductionMean, EmptySliceProducesNaN) {
    Tensor x({0, 3}, dtype::float32, Device::cpu());
    auto y = mean(x, {0});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{3}));
    for (float v : read_all<float>(y)) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(ReductionProd, EmptySliceProducesOne) {
    Tensor x({0, 3}, dtype::float32, Device::cpu());
    auto y = prod(x, {0});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{3}));
    for (float v : read_all<float>(y)) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

TEST(ReductionSum, RejectsBfloat16) {
    Tensor x({4}, dtype::bfloat16, Device::cpu());
    EXPECT_THROW(sum(x), DTypeError);
    EXPECT_THROW(prod(x), DTypeError);
    EXPECT_THROW(mean(x), DTypeError);
}

TEST(ReductionSum, OutOfRangeAxisThrows) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(sum(x, {2}), ShapeError);
    EXPECT_THROW(sum(x, {0, 0}), ShapeError);
}

// ---------- max / min ----------

TEST(ReductionMax, WholeTensorFloat32) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 5, 2, 4, 3, 0});
    auto y = max(x);
    EXPECT_EQ(y.dtype(), dtype::float32);
    EXPECT_TRUE(y.shape().empty());
    EXPECT_FLOAT_EQ(read_all<float>(y)[0], 5.0f);
}

TEST(ReductionMin, WholeTensorPreservesDtype) {
    auto x = make_filled<std::int32_t>({4}, dtype::int32, {3, 1, 4, 1});
    auto y = min(x);
    EXPECT_EQ(y.dtype(), dtype::int32);
    EXPECT_EQ(read_all<std::int32_t>(y)[0], 1);
}

TEST(ReductionMax, MultiAxisFloat) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 5, 2, 4, 3, 0});
    // Explicit vector form disambiguates from the single-axis
    // `max(const Tensor&, int64_t)` overload that would match `{1}`.
    auto y = max(x, std::vector<std::int64_t>{1});
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(read_all<float>(y), (std::vector<float>{5.0f, 4.0f}));
}

TEST(ReductionMax, EmptySliceThrows) {
    Tensor x({0, 3}, dtype::float32, Device::cpu());
    EXPECT_THROW(max(x, std::vector<std::int64_t>{0}), ShapeError);
    EXPECT_THROW(min(x, std::vector<std::int64_t>{0}), ShapeError);
    EXPECT_THROW(max(x, static_cast<std::int64_t>(0)), ShapeError);
    EXPECT_THROW(argmax(x, static_cast<std::int64_t>(0)), ShapeError);
}

// ---------- single-axis max with indices ----------

TEST(ReductionMax, SingleAxisReturnsValuesAndIndices) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 5, 2, 4, 3, 0});
    auto vi = max(x, /*dim=*/1);
    EXPECT_EQ(vi.values.dtype(), dtype::float32);
    EXPECT_EQ(vi.indices.dtype(), dtype::int64);
    EXPECT_EQ(vi.values.shape(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(vi.indices.shape(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(read_all<float>(vi.values), (std::vector<float>{5.0f, 4.0f}));
    EXPECT_EQ(read_all<std::int64_t>(vi.indices), (std::vector<std::int64_t>{1, 0}));
}

TEST(ReductionMax, SingleAxisKeepdim) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 5, 2, 4, 3, 0});
    auto vi = max(x, /*dim=*/-1, /*keepdim=*/true);
    EXPECT_EQ(vi.values.shape(), (std::vector<std::int64_t>{2, 1}));
    EXPECT_EQ(vi.indices.shape(), (std::vector<std::int64_t>{2, 1}));
}

TEST(ReductionMin, SingleAxisInt32) {
    auto x = make_filled<std::int32_t>({3, 2}, dtype::int32, {3, 1, 4, 1, 5, 9});
    auto vi = min(x, /*dim=*/0);
    EXPECT_EQ(vi.values.dtype(), dtype::int32);
    EXPECT_EQ(read_all<std::int32_t>(vi.values), (std::vector<std::int32_t>{3, 1}));
    EXPECT_EQ(read_all<std::int64_t>(vi.indices), (std::vector<std::int64_t>{0, 0}));
}

// ---------- argmax / argmin ----------

TEST(ReductionArgmax, AlwaysInt64) {
    auto x = make_filled<float>({2, 3}, dtype::float32, {1, 5, 2, 4, 3, 0});
    auto y = argmax(x, /*dim=*/1);
    EXPECT_EQ(y.dtype(), dtype::int64);
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(read_all<std::int64_t>(y), (std::vector<std::int64_t>{1, 0}));
}

TEST(ReductionArgmax, FirstOccurrenceWinsOnTie) {
    // Tied input: argmax should return the *first* index of the
    // maximum value (matches PyTorch).
    auto x = make_filled<float>({5}, dtype::float32, {1.0f, 3.0f, 3.0f, 2.0f, 3.0f});
    auto y = argmax(x, /*dim=*/0);
    EXPECT_EQ(read_all<std::int64_t>(y)[0], 1);
}

TEST(ReductionArgmin, NegativeAxisAndKeepdim) {
    auto x = make_filled<std::int32_t>({2, 3}, dtype::int32, {3, 1, 4, 1, 5, 9});
    auto y = argmin(x, /*dim=*/-1, /*keepdim=*/true);
    EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{2, 1}));
    EXPECT_EQ(read_all<std::int64_t>(y), (std::vector<std::int64_t>{1, 0}));
}

TEST(ReductionMax, NaNPropagatesAndArgmaxReportsItsIndex) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto x = make_filled<float>({4}, dtype::float32, {1.0f, nan, 2.0f, 0.5f});
    auto vi = max(x, /*dim=*/0);
    EXPECT_TRUE(std::isnan(read_all<float>(vi.values)[0]));
    EXPECT_EQ(read_all<std::int64_t>(vi.indices)[0], 1);
}

TEST(ReductionArgmax, RejectsZeroDimTensor) {
    Tensor x({}, dtype::float32, Device::cpu());
    EXPECT_THROW(argmax(x, 0), ShapeError);
}
