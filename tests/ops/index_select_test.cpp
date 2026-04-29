//===- tests/ops/index_select_test.cpp - functional gather tests ----------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functional tests for `ctorch::index_select`. Issue #10: gather rows
/// of a source tensor along a single axis according to an integer index
/// tensor; produces a fresh contiguous output.
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
using ctorch::DTypeError;
using ctorch::dtype;
using ctorch::index_select;
using ctorch::ShapeError;
using ctorch::Tensor;

namespace {

void fill_iota_float(Tensor& t) {
    auto* p = static_cast<float*>(t.storage().data());
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i);
    }
}

void set_indices64(Tensor& t, std::initializer_list<std::int64_t> values) {
    auto* p = static_cast<std::int64_t*>(t.storage().data());
    std::int64_t i = 0;
    for (auto v : values) {
        p[i++] = v;
    }
}

void set_indices32(Tensor& t, std::initializer_list<std::int32_t> values) {
    auto* p = static_cast<std::int32_t*>(t.storage().data());
    std::int64_t i = 0;
    for (auto v : values) {
        p[i++] = v;
    }
}

const float* fdata(const Tensor& t) {
    return static_cast<const float*>(t.storage().data()) + t.offset();
}

} // namespace

TEST(IndexSelect, GatherRowsAlongLeadingAxis) {
    // src shape (4, 3): row r contains {3r, 3r+1, 3r+2}.
    Tensor src({4, 3}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({3}, dtype::int64, Device::cpu());
    set_indices64(idx, {2, 0, 3});

    Tensor out = index_select(src, /*dim=*/0, idx);

    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({3, 3}));
    EXPECT_EQ(out.dtype(), dtype::float32);
    EXPECT_TRUE(out.is_contiguous());

    const auto* p = fdata(out);
    // Row 0 from src row 2: {6, 7, 8}.
    EXPECT_EQ(p[0], 6.0f);
    EXPECT_EQ(p[2], 8.0f);
    // Row 1 from src row 0: {0, 1, 2}.
    EXPECT_EQ(p[3], 0.0f);
    EXPECT_EQ(p[5], 2.0f);
    // Row 2 from src row 3: {9, 10, 11}.
    EXPECT_EQ(p[6], 9.0f);
    EXPECT_EQ(p[8], 11.0f);
}

TEST(IndexSelect, GatherAlongInnerAxis) {
    // src shape (3, 4): row r contains {4r, 4r+1, 4r+2, 4r+3}.
    Tensor src({3, 4}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({2}, dtype::int64, Device::cpu());
    set_indices64(idx, {3, 1});

    Tensor out = index_select(src, /*dim=*/1, idx);

    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({3, 2}));
    const auto* p = fdata(out);
    // Per-row gather of col 3 then col 1.
    EXPECT_EQ(p[0], 3.0f);
    EXPECT_EQ(p[1], 1.0f);
    EXPECT_EQ(p[2], 7.0f);
    EXPECT_EQ(p[3], 5.0f);
    EXPECT_EQ(p[4], 11.0f);
    EXPECT_EQ(p[5], 9.0f);
}

TEST(IndexSelect, NegativeDimIsNormalised) {
    Tensor src({3, 4}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({2}, dtype::int64, Device::cpu());
    set_indices64(idx, {0, 2});

    Tensor a = index_select(src, /*dim=*/-1, idx);
    Tensor b = index_select(src, /*dim=*/1, idx);

    ASSERT_EQ(a.shape(), b.shape());
    const auto* pa = fdata(a);
    const auto* pb = fdata(b);
    for (std::int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(pa[i], pb[i]) << "i=" << i;
    }
}

TEST(IndexSelect, EmptyIndexProducesZeroAlongDim) {
    Tensor src({3, 4}, dtype::float32, Device::cpu());
    Tensor idx({0}, dtype::int64, Device::cpu());

    Tensor out = index_select(src, /*dim=*/0, idx);
    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({0, 4}));
    EXPECT_EQ(out.numel(), 0);
}

TEST(IndexSelect, OutputIsAFreshContiguousAllocation) {
    Tensor src({4, 3}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({2}, dtype::int64, Device::cpu());
    set_indices64(idx, {0, 1});

    Tensor out = index_select(src, 0, idx);
    EXPECT_NE(out.storage().data(), src.storage().data());
    EXPECT_TRUE(out.is_contiguous());
}

TEST(IndexSelect, AcceptsInt32Indices) {
    Tensor src({4, 3}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({2}, dtype::int32, Device::cpu());
    set_indices32(idx, {3, 1});

    Tensor out = index_select(src, 0, idx);
    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({2, 3}));
    const auto* p = fdata(out);
    EXPECT_EQ(p[0], 9.0f);
    EXPECT_EQ(p[3], 3.0f);
}

TEST(IndexSelect, NegativeIndexIsNormalised) {
    Tensor src({4, 3}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({2}, dtype::int64, Device::cpu());
    set_indices64(idx, {-1, -4});

    Tensor out = index_select(src, 0, idx);
    const auto* p = fdata(out);
    // -1 → row 3 = {9,10,11}; -4 → row 0 = {0,1,2}.
    EXPECT_EQ(p[0], 9.0f);
    EXPECT_EQ(p[3], 0.0f);
}

TEST(IndexSelect, OutOfRangeIndexThrows) {
    Tensor src({4, 3}, dtype::float32, Device::cpu());
    Tensor idx({1}, dtype::int64, Device::cpu());
    set_indices64(idx, {7});
    EXPECT_THROW((void)index_select(src, 0, idx), ShapeError);

    set_indices64(idx, {-100});
    EXPECT_THROW((void)index_select(src, 0, idx), ShapeError);
}

TEST(IndexSelect, GatherFrom3DSource) {
    // src shape (2, 3, 4). Use innermost axis (dim=2) to verify rank-3
    // shape arithmetic.
    Tensor src({2, 3, 4}, dtype::float32, Device::cpu());
    fill_iota_float(src);

    Tensor idx({3}, dtype::int64, Device::cpu());
    set_indices64(idx, {2, 0, 3});

    Tensor out = index_select(src, /*dim=*/2, idx);
    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({2, 3, 3}));
    const auto* p = fdata(out);
    // For (b=0, r=0): src[(0,0,2)] = 2, src[(0,0,0)] = 0, src[(0,0,3)] = 3.
    EXPECT_EQ(p[0], 2.0f);
    EXPECT_EQ(p[1], 0.0f);
    EXPECT_EQ(p[2], 3.0f);
    // For (b=1, r=2): linear input start = 1*12 + 2*4 = 20, so we expect
    // {22, 20, 23}.
    EXPECT_EQ(p[15], 22.0f);
    EXPECT_EQ(p[16], 20.0f);
    EXPECT_EQ(p[17], 23.0f);
}

TEST(IndexSelect, FromStridedSource) {
    // Verify the kernel honours the source strides — gather from a
    // permuted (non-contiguous) view.
    Tensor src({3, 4}, dtype::float32, Device::cpu());
    fill_iota_float(src);
    Tensor t = src.permute({1, 0}); // shape (4,3), non-contiguous

    Tensor idx({2}, dtype::int64, Device::cpu());
    set_indices64(idx, {0, 2});

    Tensor out = index_select(t, /*dim=*/0, idx);
    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({2, 3}));
    EXPECT_TRUE(out.is_contiguous());
    const auto* p = fdata(out);
    // t row 0 = src col 0 = {0, 4, 8}; t row 2 = src col 2 = {2, 6, 10}.
    EXPECT_EQ(p[0], 0.0f);
    EXPECT_EQ(p[1], 4.0f);
    EXPECT_EQ(p[2], 8.0f);
    EXPECT_EQ(p[3], 2.0f);
    EXPECT_EQ(p[4], 6.0f);
    EXPECT_EQ(p[5], 10.0f);
}
