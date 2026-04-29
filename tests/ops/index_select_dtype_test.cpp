//===- tests/ops/index_select_dtype_test.cpp - dtype tests ---------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// dtype-related coverage for `ctorch::index_select` — int32/int64 index
/// support, rejection of float / bool / bfloat16 indices, dim-out-of-range
/// errors, and rank-0 source rejection.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/indexing.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::index_select;
using ctorch::ShapeError;
using ctorch::Tensor;

TEST(IndexSelectDType, SourceDtypePreservedInOutput) {
    Tensor src_f32({3}, dtype::float32, Device::cpu());
    Tensor src_f64({3}, dtype::float64, Device::cpu());
    Tensor src_i32({3}, dtype::int32, Device::cpu());
    Tensor src_i64({3}, dtype::int64, Device::cpu());
    Tensor idx({1}, dtype::int64, Device::cpu());

    EXPECT_EQ(index_select(src_f32, 0, idx).dtype(), dtype::float32);
    EXPECT_EQ(index_select(src_f64, 0, idx).dtype(), dtype::float64);
    EXPECT_EQ(index_select(src_i32, 0, idx).dtype(), dtype::int32);
    EXPECT_EQ(index_select(src_i64, 0, idx).dtype(), dtype::int64);
}

TEST(IndexSelectDType, RejectsFloatIndices) {
    Tensor src({3}, dtype::float32, Device::cpu());
    Tensor idx({1}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)index_select(src, 0, idx), DTypeError);
}

TEST(IndexSelectDType, RejectsBoolIndices) {
    Tensor src({3}, dtype::float32, Device::cpu());
    Tensor idx({1}, dtype::bool_, Device::cpu());
    EXPECT_THROW((void)index_select(src, 0, idx), DTypeError);
}

TEST(IndexSelectDType, RejectsRankNeOneIndices) {
    Tensor src({3}, dtype::float32, Device::cpu());
    Tensor idx2d({1, 1}, dtype::int64, Device::cpu());
    EXPECT_THROW((void)index_select(src, 0, idx2d), ShapeError);

    Tensor idx0d({}, dtype::int64, Device::cpu());
    EXPECT_THROW((void)index_select(src, 0, idx0d), ShapeError);
}

TEST(IndexSelectDType, DimOutOfRangeThrows) {
    Tensor src({3, 4}, dtype::float32, Device::cpu());
    Tensor idx({1}, dtype::int64, Device::cpu());
    EXPECT_THROW((void)index_select(src, 5, idx), ShapeError);
    EXPECT_THROW((void)index_select(src, -3, idx), ShapeError);
}

TEST(IndexSelectDType, RankZeroSourceRejected) {
    Tensor src({}, dtype::float32, Device::cpu());
    Tensor idx({1}, dtype::int64, Device::cpu());
    EXPECT_THROW((void)index_select(src, 0, idx), ShapeError);
}

TEST(IndexSelectDType, BFloat16SourceRejected) {
    Tensor src({3}, dtype::bfloat16, Device::cpu());
    Tensor idx({1}, dtype::int64, Device::cpu());
    EXPECT_THROW((void)index_select(src, 0, idx), DTypeError);
}

TEST(IndexSelectDType, CpuAcceptsRankAboveKMaxRank) {
    // The CUDA plan caps shape/stride at kMaxRank=16, but the CPU backend
    // uses std::vector-backed coords and handles arbitrary rank. Guard
    // against accidentally re-introducing a shared front-door rank cap.
    std::vector<std::int64_t> shape(17, 1);
    Tensor src(shape, dtype::float32, Device::cpu());
    Tensor idx({1}, dtype::int64, Device::cpu());
    Tensor out = index_select(src, 0, idx);
    EXPECT_EQ(out.shape().size(), 17u);
}
