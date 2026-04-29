//===- tests/tensor/slice_test.cpp - slice / select / narrow tests --------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// View semantics for `Tensor::slice`, `Tensor::select`, and
/// `Tensor::narrow`. Issue #10: zero-copy view-producing ops with shared
/// storage, normalised negative bounds, and validated step / index ranges.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::ShapeError;
using ctorch::Tensor;

namespace {

void fill_iota_int32(Tensor& t) {
    auto* p = static_cast<std::int32_t*>(t.storage().data());
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<std::int32_t>(i);
    }
}

const std::int32_t* idata(const Tensor& t) {
    return static_cast<const std::int32_t*>(t.storage().data()) + t.offset();
}

} // namespace

TEST(TensorSlice, BasicSliceProducesViewSharingStorage) {
    Tensor t({4, 5}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor s = t.slice(/*dim=*/0, /*start=*/1, /*end=*/3);

    EXPECT_EQ(s.storage().data(), t.storage().data());
    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({2, 5}));
    EXPECT_EQ(s.stride(), std::vector<std::int64_t>({5, 1}));
    EXPECT_EQ(s.offset(), 5);

    // Row 1 = {5,6,7,8,9}; Row 2 = {10,11,12,13,14}.
    const auto* p = idata(s);
    EXPECT_EQ(p[0], 5);
    EXPECT_EQ(p[4], 9);
    EXPECT_EQ(p[5], 10);
    EXPECT_EQ(p[9], 14);
}

TEST(TensorSlice, SliceWithStepStridesSourceAxis) {
    Tensor t({6}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor s = t.slice(0, 0, 6, 2);

    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({3}));
    EXPECT_EQ(s.stride(), std::vector<std::int64_t>({2}));

    // Logical view = {0, 2, 4}; honour stride[0] = 2 when reading.
    const auto* p = idata(s);
    EXPECT_EQ(p[0 * 2], 0);
    EXPECT_EQ(p[1 * 2], 2);
    EXPECT_EQ(p[2 * 2], 4);
}

TEST(TensorSlice, NegativeStartAndEndAreNormalised) {
    Tensor t({5}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor s = t.slice(0, -3, -1);

    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({2}));
    const auto* p = idata(s);
    EXPECT_EQ(p[0], 2);
    EXPECT_EQ(p[1], 3);
}

TEST(TensorSlice, NegativeDimIsNormalised) {
    Tensor t({2, 4}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor s = t.slice(/*dim=*/-1, /*start=*/1, /*end=*/3);

    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({2, 2}));
    EXPECT_EQ(s.stride(), std::vector<std::int64_t>({4, 1}));
    EXPECT_EQ(s.offset(), 1);
}

TEST(TensorSlice, OutOfRangeBoundsAreClampedNotThrown) {
    // PyTorch-compatible: start/end clamp to [0, size]; the result is just
    // the valid sub-range.
    Tensor t({4}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor s = t.slice(0, -100, 100);
    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({4}));
}

TEST(TensorSlice, EmptySliceProducesZeroSizedTensor) {
    Tensor t({4}, dtype::int32, Device::cpu());

    Tensor s = t.slice(0, 2, 2);
    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({0}));
    EXPECT_EQ(s.numel(), 0);
}

TEST(TensorSlice, NonPositiveStepThrows) {
    Tensor t({4}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.slice(0, 0, 4, 0), ShapeError);
    EXPECT_THROW((void)t.slice(0, 0, 4, -1), ShapeError);
}

TEST(TensorSlice, RankZeroRejected) {
    Tensor t({}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.slice(0, 0, 0), ShapeError);
}

TEST(TensorSelect, DropsDimAndSharesStorage) {
    Tensor t({3, 4}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor row = t.select(/*dim=*/0, /*index=*/1);

    EXPECT_EQ(row.storage().data(), t.storage().data());
    EXPECT_EQ(row.shape(), std::vector<std::int64_t>({4}));
    EXPECT_EQ(row.stride(), std::vector<std::int64_t>({1}));
    EXPECT_EQ(row.offset(), 4);

    const auto* p = idata(row);
    EXPECT_EQ(p[0], 4);
    EXPECT_EQ(p[3], 7);
}

TEST(TensorSelect, RankOneToRankZero) {
    Tensor t({5}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor s = t.select(0, 2);

    EXPECT_TRUE(s.shape().empty());
    EXPECT_EQ(s.numel(), 1);
    const auto* p = idata(s);
    EXPECT_EQ(p[0], 2);
}

TEST(TensorSelect, NegativeIndexIsNormalised) {
    Tensor t({4}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor last = t.select(0, -1);
    const auto* p = idata(last);
    EXPECT_EQ(p[0], 3);
}

TEST(TensorSelect, OutOfRangeIndexThrows) {
    Tensor t({4}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.select(0, 4), ShapeError);
    EXPECT_THROW((void)t.select(0, -5), ShapeError);
}

TEST(TensorSelect, Int64MinIndexThrowsWithoutOverflow) {
    // `index + size` would overflow signed-64 (UB) for INT64_MIN — the
    // bounds check has to fire before normalising.
    Tensor t({4}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.select(0, std::numeric_limits<std::int64_t>::min()), ShapeError);
}

TEST(TensorSlice, ExtremeNegativeBoundsClampWithoutOverflow) {
    // INT64_MIN in start / end must clamp safely instead of overflowing.
    Tensor t({4}, dtype::int32, Device::cpu());
    Tensor s = t.slice(0, std::numeric_limits<std::int64_t>::min(), 100);
    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({4}));
}

TEST(TensorSlice, HugeStepLengthComputationDoesNotOverflow) {
    // With step == INT64_MAX the original `(span + step - 1)` ceil-div
    // formula would overflow signed-64. Result should still be a valid
    // length-1 slice of the first element.
    Tensor t({4}, dtype::int32, Device::cpu());
    Tensor s = t.slice(0, 0, 4, std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(s.shape(), std::vector<std::int64_t>({1}));
}

TEST(TensorNarrow, IsSugarForSliceWithStepOne) {
    Tensor t({5}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor n = t.narrow(0, 1, 3);

    EXPECT_EQ(n.shape(), std::vector<std::int64_t>({3}));
    const auto* p = idata(n);
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[2], 3);
}

TEST(TensorNarrow, NegativeStartIsNormalised) {
    Tensor t({5}, dtype::int32, Device::cpu());
    fill_iota_int32(t);

    Tensor n = t.narrow(0, -3, 2);

    EXPECT_EQ(n.shape(), std::vector<std::int64_t>({2}));
    const auto* p = idata(n);
    EXPECT_EQ(p[0], 2);
    EXPECT_EQ(p[1], 3);
}

TEST(TensorNarrow, OutOfRangeRangeThrows) {
    Tensor t({4}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.narrow(0, 0, 5), ShapeError);
    EXPECT_THROW((void)t.narrow(0, 3, 2), ShapeError);
    EXPECT_THROW((void)t.narrow(0, 0, -1), ShapeError);
}

TEST(TensorNarrow, Int64MinStartThrowsWithoutOverflow) {
    Tensor t({4}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.narrow(0, std::numeric_limits<std::int64_t>::min(), 1), ShapeError);
}

TEST(TensorNarrow, HugeLengthThrowsWithoutAdditiveOverflow) {
    // `adj_start + length > size` would overflow signed-64 for length
    // near INT64_MAX. The subtraction-based check should still surface
    // the OOB condition as ShapeError.
    Tensor t({4}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)t.narrow(0, 0, std::numeric_limits<std::int64_t>::max()), ShapeError);
    EXPECT_THROW((void)t.narrow(0, 1, std::numeric_limits<std::int64_t>::max()), ShapeError);
}

TEST(TensorViewOps, StorageUseCountReflectsSharing) {
    Tensor t({3, 4}, dtype::int32, Device::cpu());
    EXPECT_EQ(t.storage().use_count(), 1);

    Tensor s = t.slice(0, 1, 2);
    EXPECT_EQ(t.storage().use_count(), 2);

    Tensor r = t.select(0, 0);
    EXPECT_EQ(t.storage().use_count(), 3);

    Tensor n = t.narrow(1, 0, 2);
    EXPECT_EQ(t.storage().use_count(), 4);
}
