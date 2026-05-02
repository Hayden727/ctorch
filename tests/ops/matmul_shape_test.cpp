//===- tests/ops/matmul_shape_test.cpp - matmul shape rules ---------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functional + shape tests for the five PyTorch `matmul` cases:
/// 1-D × 1-D, 1-D × 2-D, 2-D × 1-D, 2-D × 2-D, batched. Each case uses
/// small, hand-computed reference values so the test fails loudly if
/// the BLAS wiring goes off.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/linalg.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

using ctorch::Device;
using ctorch::dtype;
using ctorch::matmul;
using ctorch::ShapeError;
using ctorch::Tensor;

namespace {

Tensor make_f32(std::vector<std::int64_t> shape, std::initializer_list<float> values) {
    Tensor t(shape, dtype::float32, Device::cpu());
    auto* p = static_cast<float*>(t.storage().data());
    std::int64_t i = 0;
    for (float v : values) {
        p[i++] = v;
    }
    return t;
}

const float* fdata(const Tensor& t) {
    return static_cast<const float*>(t.storage().data()) + t.offset();
}

} // namespace

TEST(MatmulShape, OneDByOneDProducesScalar) {
    Tensor a = make_f32({3}, {1.0f, 2.0f, 3.0f});
    Tensor b = make_f32({3}, {4.0f, 5.0f, 6.0f});
    Tensor c = matmul(a, b);
    EXPECT_TRUE(c.shape().empty());
    EXPECT_EQ(c.numel(), 1);
    EXPECT_FLOAT_EQ(fdata(c)[0], 1 * 4 + 2 * 5 + 3 * 6);
}

TEST(MatmulShape, OneDByTwoDReturns1D) {
    // a (K=3) × b (3, 2) → 1-D length 2.
    Tensor a = make_f32({3}, {1.0f, 2.0f, 3.0f});
    Tensor b = make_f32({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2}));
    // (1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6) = (22, 28)
    EXPECT_FLOAT_EQ(fdata(c)[0], 22.0f);
    EXPECT_FLOAT_EQ(fdata(c)[1], 28.0f);
}

TEST(MatmulShape, TwoDByOneDReturns1D) {
    // a (2, 3) × b (3,) → 1-D length 2.
    Tensor a = make_f32({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = make_f32({3}, {1.0f, 2.0f, 3.0f});
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2}));
    EXPECT_FLOAT_EQ(fdata(c)[0], 1 + 4 + 9);   // 14
    EXPECT_FLOAT_EQ(fdata(c)[1], 4 + 10 + 18); // 32
}

TEST(MatmulShape, TwoDByTwoDStandardGemm) {
    // (2, 3) × (3, 2) → (2, 2)
    Tensor a = make_f32({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = make_f32({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2, 2}));
    // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //  = [[58, 64], [139, 154]]
    EXPECT_FLOAT_EQ(fdata(c)[0], 58.0f);
    EXPECT_FLOAT_EQ(fdata(c)[1], 64.0f);
    EXPECT_FLOAT_EQ(fdata(c)[2], 139.0f);
    EXPECT_FLOAT_EQ(fdata(c)[3], 154.0f);
}

TEST(MatmulShape, BatchedThreeDByThreeD) {
    // a (2, 2, 3) × b (2, 3, 2) → (2, 2, 2)
    Tensor a = make_f32({2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0});
    Tensor b = make_f32({2, 3, 2}, {1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0});
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2, 2, 2}));
    // Batch 0: identity-like → [[1, 2], [4, 5]]
    EXPECT_FLOAT_EQ(fdata(c)[0], 1.0f);
    EXPECT_FLOAT_EQ(fdata(c)[1], 2.0f);
    EXPECT_FLOAT_EQ(fdata(c)[2], 4.0f);
    EXPECT_FLOAT_EQ(fdata(c)[3], 5.0f);
    // Batch 1: a=[[1,0,0],[0,1,0]], b=[[0,1],[1,0],[0,0]] → [[0,1],[1,0]]
    EXPECT_FLOAT_EQ(fdata(c)[4], 0.0f);
    EXPECT_FLOAT_EQ(fdata(c)[5], 1.0f);
    EXPECT_FLOAT_EQ(fdata(c)[6], 1.0f);
    EXPECT_FLOAT_EQ(fdata(c)[7], 0.0f);
}

TEST(MatmulShape, BatchedBroadcastLeadingDim) {
    // a (1, 2, 3) × b (4, 3, 2) → (4, 2, 2)
    Tensor a = make_f32({1, 2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b(/*shape=*/{4, 3, 2}, dtype::float32, Device::cpu());
    auto* bp = static_cast<float*>(b.storage().data());
    for (std::int64_t i = 0; i < b.numel(); ++i) {
        bp[i] = static_cast<float>(i % 7);
    }
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({4, 2, 2}));
    // Sanity: c is non-empty, no shape error thrown.
    EXPECT_GT(c.numel(), 0);
}

TEST(MatmulShape, InnerDimMismatchThrows) {
    Tensor a = make_f32({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = make_f32({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW((void)matmul(a, b), ShapeError);
}

TEST(MatmulShape, ZeroDInputRejected) {
    Tensor a({}, dtype::float32, Device::cpu());
    Tensor b = make_f32({3}, {1, 2, 3});
    EXPECT_THROW((void)matmul(a, b), ShapeError);
    EXPECT_THROW((void)matmul(b, a), ShapeError);
}

TEST(MatmulShape, ZeroSizedBatchBroadcastIsPreserved) {
    // Broadcasting a 0-sized batch dim against a 1-sized one must
    // collapse to 0, not 1 — otherwise the planner would schedule
    // GEMMs against empty storage. Result is a valid empty tensor.
    Tensor a({0, 2, 3}, dtype::float32, Device::cpu());
    Tensor b({1, 3, 4}, dtype::float32, Device::cpu());
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({0, 2, 4}));
    EXPECT_EQ(c.numel(), 0);
}

TEST(MatmulShape, MatmulOfTransposeProducesGramMatrix) {
    // c = a · a^T must be symmetric and equal a a^T element-wise.
    Tensor a = make_f32({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor c = matmul(a, a.T());
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2, 2}));
    EXPECT_FLOAT_EQ(fdata(c)[0], 1 + 4 + 9);    // 14
    EXPECT_FLOAT_EQ(fdata(c)[1], 4 + 10 + 18);  // 32
    EXPECT_FLOAT_EQ(fdata(c)[2], 4 + 10 + 18);  // 32
    EXPECT_FLOAT_EQ(fdata(c)[3], 16 + 25 + 36); // 77
}
