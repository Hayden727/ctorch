//===- tests/tensor/tensor_view_test.cpp - View / permute / contiguous ----===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Exercises Tensor construction, default-stride layout, view/reshape/
/// permute/contiguous, and the zero-init guarantee from §AC1 / §AC3.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::Tensor;

namespace {

void fill_iota_float(Tensor& t) {
    auto* p = static_cast<float*>(t.storage().data());
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i);
    }
}

} // namespace

TEST(Tensor, ConstructAllocatesAndZeroFills) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());

    EXPECT_EQ(t.shape(), std::vector<std::int64_t>({2, 3}));
    EXPECT_EQ(t.stride(), std::vector<std::int64_t>({3, 1}));
    EXPECT_EQ(t.offset(), 0);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.dtype(), dtype::float32);
    EXPECT_TRUE(t.device().is_cpu());

    const auto* p = static_cast<const float*>(t.storage().data());
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(p[i], 0.0f) << "element " << i;
    }
}

TEST(Tensor, ViewSharesStorageAndRecomputesStride) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota_float(t);

    Tensor v = t.view({6});

    EXPECT_EQ(t.storage().data(), v.storage().data());
    EXPECT_EQ(v.shape(), std::vector<std::int64_t>({6}));
    EXPECT_EQ(v.stride(), std::vector<std::int64_t>({1}));
    EXPECT_EQ(v.numel(), 6);

    const auto* vp = static_cast<const float*>(v.storage().data());
    for (std::int64_t i = 0; i < v.numel(); ++i) {
        EXPECT_EQ(vp[i], static_cast<float>(i));
    }
}

TEST(Tensor, PermuteRearrangesShapeAndStrideWithoutCopy) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota_float(t);

    Tensor p = t.permute({1, 0});

    EXPECT_EQ(t.storage().data(), p.storage().data());
    EXPECT_EQ(p.shape(), std::vector<std::int64_t>({3, 2}));
    EXPECT_EQ(p.stride(), std::vector<std::int64_t>({1, 3}));
    EXPECT_FALSE(p.is_contiguous());
}

TEST(Tensor, ContiguousAfterPermuteAllocatesFreshStorageWithRowMajorStride) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota_float(t);

    Tensor p = t.permute({1, 0});
    Tensor c = p.contiguous();

    EXPECT_NE(t.storage().data(), c.storage().data());
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({3, 2}));
    EXPECT_EQ(c.stride(), std::vector<std::int64_t>({2, 1}));
    EXPECT_TRUE(c.is_contiguous());

    // Logical layout of `p` is `[[0,3],[1,4],[2,5]]`. After contiguous() the
    // bytes match that order in memory.
    const auto* cp = static_cast<const float*>(c.storage().data());
    EXPECT_EQ(cp[0], 0.0f);
    EXPECT_EQ(cp[1], 3.0f);
    EXPECT_EQ(cp[2], 1.0f);
    EXPECT_EQ(cp[3], 4.0f);
    EXPECT_EQ(cp[4], 2.0f);
    EXPECT_EQ(cp[5], 5.0f);
}

TEST(Tensor, ReshapeFallsBackToContiguousForNonContiguousInputs) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    fill_iota_float(t);

    Tensor p = t.permute({1, 0}); // non-contiguous
    Tensor r = p.reshape({6});    // must copy

    EXPECT_NE(t.storage().data(), r.storage().data());
    EXPECT_EQ(r.shape(), std::vector<std::int64_t>({6}));
    EXPECT_TRUE(r.is_contiguous());
}

TEST(Tensor, ViewOnNonContiguousThrows) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    Tensor p = t.permute({1, 0});
    EXPECT_THROW((void)p.view({6}), std::runtime_error);
}

TEST(Tensor, ViewWithMismatchedNumelThrows) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)t.view({2, 4}), std::runtime_error);
}

TEST(Tensor, PermuteRejectsBadDims) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)t.permute({0, 0}), std::runtime_error);
    EXPECT_THROW((void)t.permute({0}), std::runtime_error);
}
