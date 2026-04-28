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
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::mean;
using ctorch::prod;
using ctorch::ShapeError;
using ctorch::sum;
using ctorch::Tensor;

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
