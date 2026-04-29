//===- tests/ops/transpose_test.cpp - transpose / T() tests ---------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// View semantics for `ctorch::transpose(x, i, j)` and `Tensor::T()`.
/// Both must be zero-copy and equivalent to a permute that swaps the
/// chosen pair of axes.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/linalg.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::ShapeError;
using ctorch::Tensor;
using ctorch::transpose;

namespace {

void fill_iota(Tensor& t) {
    auto* p = static_cast<float*>(t.storage().data());
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i);
    }
}

} // namespace

TEST(Transpose, TwoDSwapsAxesAndSharesStorage) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota(t);

    Tensor tr = transpose(t, 0, 1);

    EXPECT_EQ(tr.storage().data(), t.storage().data());
    EXPECT_EQ(tr.shape(), std::vector<std::int64_t>({3, 2}));
    EXPECT_EQ(tr.stride(), std::vector<std::int64_t>({1, 3}));
    EXPECT_FALSE(tr.is_contiguous());
}

TEST(Transpose, NegativeDimsAreNormalised) {
    Tensor t({2, 3, 4}, dtype::float32, Device::cpu());
    Tensor a = transpose(t, -1, -2);
    Tensor b = transpose(t, 2, 1);
    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.stride(), b.stride());
}

TEST(Transpose, EqualDimsReturnsInputUnchanged) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    Tensor r = transpose(t, 0, 0);
    EXPECT_EQ(r.shape(), t.shape());
    EXPECT_EQ(r.stride(), t.stride());
}

TEST(Transpose, OutOfRangeDimThrows) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)transpose(t, 0, 5), ShapeError);
    EXPECT_THROW((void)transpose(t, -3, 1), ShapeError);
}

TEST(Transpose, RankZeroRejected) {
    Tensor t({}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)transpose(t, 0, 0), ShapeError);
}

TEST(TensorT, ShorthandFor2DTranspose) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota(t);

    Tensor tr = t.T();
    Tensor expected = transpose(t, 0, 1);

    EXPECT_EQ(tr.shape(), expected.shape());
    EXPECT_EQ(tr.stride(), expected.stride());
    EXPECT_EQ(tr.storage().data(), t.storage().data());
}

TEST(TensorT, NonTwoDInputThrows) {
    Tensor t1d({4}, dtype::float32, Device::cpu());
    Tensor t3d({2, 3, 4}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)t1d.T(), ShapeError);
    EXPECT_THROW((void)t3d.T(), ShapeError);
}

TEST(Transpose, MutationViaViewVisibleInSource) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota(t);
    Tensor tr = transpose(t, 0, 1);

    // tr is shape (3, 2) with stride (1, 3). Writing to tr[0, 1] should
    // land at storage element t[1, 0] = index 3.
    auto* trp = static_cast<float*>(tr.storage().data()) + tr.offset();
    trp[0 * 1 + 1 * 3] = 99.0f;

    const auto* tp = static_cast<const float*>(t.storage().data());
    EXPECT_EQ(tp[3], 99.0f);
}
