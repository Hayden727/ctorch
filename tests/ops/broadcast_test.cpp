//===- tests/ops/broadcast_test.cpp - broadcasting algorithm ---*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/tensor.h"
#include "ops/broadcast.h"
#include "ops/tensor_iter.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::ShapeError;
using ctorch::Tensor;
using ctorch::ops::broadcast_two;
using ctorch::ops::BroadcastResult;
using ctorch::ops::may_overlap;

namespace {

Tensor make(std::vector<std::int64_t> shape) {
    return Tensor(std::move(shape), dtype::float32, Device::cpu());
}

} // namespace

TEST(Broadcast, EqualShapesPreserveStrides) {
    auto a = make({3, 4});
    auto b = make({3, 4});
    auto br = broadcast_two(a, b);
    EXPECT_EQ(br.out_shape, std::vector<std::int64_t>({3, 4}));
    // No size-1 dim ⇒ no zero strides anywhere.
    EXPECT_EQ(br.a_stride, a.stride());
    EXPECT_EQ(br.b_stride, b.stride());
}

TEST(Broadcast, ScalarPlusVectorPadsLeadingOne) {
    auto a = make({});       // 0-d scalar
    auto b = make({5});
    auto br = broadcast_two(a, b);
    EXPECT_EQ(br.out_shape, std::vector<std::int64_t>({5}));
    EXPECT_EQ(br.a_stride, std::vector<std::int64_t>({0}));
    EXPECT_EQ(br.b_stride, b.stride());
}

TEST(Broadcast, ColumnPlusRowZeroStrideTrick) {
    auto a = make({3, 1});
    auto b = make({1, 4});
    auto br = broadcast_two(a, b);
    EXPECT_EQ(br.out_shape, std::vector<std::int64_t>({3, 4}));
    // a is `{3, 1}` so its trailing dim broadcasts → stride 0 in dim 1.
    EXPECT_EQ(br.a_stride[1], 0);
    EXPECT_EQ(br.a_stride[0], a.stride()[0]);
    // b is `{1, 4}` so its leading dim broadcasts → stride 0 in dim 0.
    EXPECT_EQ(br.b_stride[0], 0);
    EXPECT_EQ(br.b_stride[1], b.stride()[1]);
}

TEST(Broadcast, MultiDimAlignsFromTheRight) {
    auto a = make({2, 3, 4});
    auto b = make({3, 4});
    auto br = broadcast_two(a, b);
    EXPECT_EQ(br.out_shape, std::vector<std::int64_t>({2, 3, 4}));
    // `b` only has dims 1 and 2 in the output; dim 0 becomes virtual leading 1.
    EXPECT_EQ(br.b_stride[0], 0);
    EXPECT_EQ(br.b_stride[1], b.stride()[0]);
    EXPECT_EQ(br.b_stride[2], b.stride()[1]);
    // `a` carries through unchanged.
    EXPECT_EQ(br.a_stride, a.stride());
}

TEST(Broadcast, IncompatibleShapesThrowShapeError) {
    auto a = make({3, 4});
    auto b = make({3, 5});
    EXPECT_THROW(broadcast_two(a, b), ShapeError);
}

TEST(AliasOverlap, SeparateStoragesNeverOverlap) {
    auto a = make({4});
    auto b = make({4});
    EXPECT_FALSE(may_overlap(a, b));
}

TEST(AliasOverlap, SameTensorOverlaps) {
    auto a = make({4});
    EXPECT_TRUE(may_overlap(a, a));
}

TEST(AliasOverlap, ReshapedViewOnSameStorageOverlaps) {
    auto a = make({2, 4});
    auto reshaped = a.view({8});
    // A view sharing the same storage and covering the same elements must
    // be detected as aliasing.
    EXPECT_TRUE(may_overlap(a, reshaped));
}

TEST(AliasOverlap, PermutedViewOnSameStorageOverlaps) {
    auto a = make({3, 4});
    auto permuted = a.permute({1, 0});
    EXPECT_TRUE(may_overlap(a, permuted));
}
