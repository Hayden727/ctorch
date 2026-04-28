//===- tests/ops/binary_test.cpp - Element-wise binary CPU ops -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/elementwise.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using ctorch::add;
using ctorch::Device;
using ctorch::DeviceError;
using ctorch::div;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::mul;
using ctorch::ShapeError;
using ctorch::sub;
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

TEST(BinaryAdd, SameShapeFloat32) {
    auto a = make_filled<float>({3}, dtype::float32, {1.0f, 2.0f, 3.0f});
    auto b = make_filled<float>({3}, dtype::float32, {4.0f, 5.0f, 6.0f});
    auto c = add(a, b);
    EXPECT_EQ(c.dtype(), dtype::float32);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({3}));
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{5.0f, 7.0f, 9.0f}));
}

TEST(BinarySub, SameShapeInt32) {
    auto a = make_filled<std::int32_t>({4}, dtype::int32, {10, 20, 30, 40});
    auto b = make_filled<std::int32_t>({4}, dtype::int32, {1, 2, 3, 4});
    auto c = sub(a, b);
    EXPECT_EQ(c.dtype(), dtype::int32);
    EXPECT_EQ(read_all<std::int32_t>(c), (std::vector<std::int32_t>{9, 18, 27, 36}));
}

TEST(BinaryMul, BroadcastColumnByRow) {
    auto a = make_filled<float>({3, 1}, dtype::float32, {1.0f, 2.0f, 3.0f});
    auto b = make_filled<float>({1, 4}, dtype::float32, {1.0f, 2.0f, 3.0f, 4.0f});
    auto c = mul(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({3, 4}));
    // Expect outer product:
    //   1 2 3 4
    //   2 4 6 8
    //   3 6 9 12
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12}));
}

TEST(BinaryDiv, FloatDivision) {
    auto a = make_filled<float>({2}, dtype::float32, {6.0f, 9.0f});
    auto b = make_filled<float>({2}, dtype::float32, {2.0f, 3.0f});
    auto c = div(a, b);
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{3.0f, 3.0f}));
}

TEST(BinaryAdd, ScalarPlusVectorBroadcasts) {
    auto a = make_filled<float>({}, dtype::float32, {10.0f});
    auto b = make_filled<float>({3}, dtype::float32, {1.0f, 2.0f, 3.0f});
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({3}));
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{11.0f, 12.0f, 13.0f}));
}

TEST(BinaryAdd, MultiDimRightAlignedBroadcast) {
    // a: (2, 3, 4) filled with consecutive ints.
    Tensor a({2, 3, 4}, dtype::int32, Device::cpu());
    auto* ap = static_cast<std::int32_t*>(a.storage().data());
    for (std::int32_t i = 0; i < 24; ++i) {
        ap[i] = i;
    }
    // b: (3, 4) filled with consecutive ints.
    Tensor b({3, 4}, dtype::int32, Device::cpu());
    auto* bp = static_cast<std::int32_t*>(b.storage().data());
    for (std::int32_t i = 0; i < 12; ++i) {
        bp[i] = i * 100;
    }
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({2, 3, 4}));
    // c[i,j,k] = a[i,j,k] + b[j,k] = (i*12+j*4+k) + (j*4+k)*100
    auto got = read_all<std::int32_t>(c);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                const int idx = i * 12 + j * 4 + k;
                EXPECT_EQ(got[idx], (i * 12 + j * 4 + k) + (j * 4 + k) * 100) << "idx " << idx;
            }
        }
    }
}

TEST(BinaryAdd, PromotesIntPlusFloat) {
    auto a = make_filled<std::int32_t>({2}, dtype::int32, {1, 2});
    auto b = make_filled<float>({2}, dtype::float32, {0.5f, 0.25f});
    auto c = add(a, b);
    EXPECT_EQ(c.dtype(), dtype::float32);
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{1.5f, 2.25f}));
}

TEST(BinaryAdd, OperatorPlusForwardsToAdd) {
    auto a = make_filled<float>({2}, dtype::float32, {1.0f, 2.0f});
    auto b = make_filled<float>({2}, dtype::float32, {10.0f, 20.0f});
    auto c = a + b;
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{11.0f, 22.0f}));
}

TEST(BinaryAdd, ShapeMismatchThrows) {
    auto a = make_filled<float>({3}, dtype::float32, {1.0f, 2.0f, 3.0f});
    auto b = make_filled<float>({4}, dtype::float32, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW(add(a, b), ShapeError);
}

TEST(BinaryAdd, RejectsBoolArithmetic) {
    auto a = make_filled<bool>({2}, dtype::bool_, {true, false});
    auto b = make_filled<bool>({2}, dtype::bool_, {true, true});
    EXPECT_THROW(add(a, b), DTypeError);
}

TEST(BinaryAdd, RejectsMixedBoolAndNumeric) {
    // The bool-rejection contract documented in docs/ops.md must trigger
    // even when the *promoted* output dtype is numeric; otherwise the
    // bool operand silently widens to 0/1 and the user gets a numeric
    // result instead of the expected DTypeError.
    auto bool_lhs = make_filled<bool>({2}, dtype::bool_, {true, false});
    auto int_rhs = make_filled<std::int32_t>({2}, dtype::int32, {3, 4});
    EXPECT_THROW(add(bool_lhs, int_rhs), DTypeError);
    EXPECT_THROW(add(int_rhs, bool_lhs), DTypeError);
    auto float_rhs = make_filled<float>({2}, dtype::float32, {0.5f, 0.25f});
    EXPECT_THROW(mul(bool_lhs, float_rhs), DTypeError);
}

TEST(BinaryAdd, HighRankTensorWorks) {
    // The fixed-size indexer caps at kMaxRank; ensure rank-9 (with most
    // dims being size 1) still goes through the strided path without
    // throwing ShapeError.
    Tensor a({1, 1, 1, 1, 1, 1, 1, 1, 4}, dtype::float32, Device::cpu());
    auto* ap = static_cast<float*>(a.storage().data());
    ap[0] = 1.0f;
    ap[1] = 2.0f;
    ap[2] = 3.0f;
    ap[3] = 4.0f;
    Tensor b({4}, dtype::float32, Device::cpu());
    auto* bp = static_cast<float*>(b.storage().data());
    bp[0] = 10.0f;
    bp[1] = 20.0f;
    bp[2] = 30.0f;
    bp[3] = 40.0f;
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), std::vector<std::int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 4}));
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(BinaryDiv, RejectsIntegerDivision) {
    auto a = make_filled<std::int32_t>({2}, dtype::int32, {6, 8});
    auto b = make_filled<std::int32_t>({2}, dtype::int32, {2, 4});
    // Integer division by zero is UB in C++; we refuse all integer div
    // up-front so legitimate user data can never crash the process.
    EXPECT_THROW(div(a, b), DTypeError);
}

TEST(BinaryAdd, NonContiguousRhsViaPermute) {
    // a: (2, 3) row-major
    auto a = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    // b: (3, 2) row-major, then permuted to (2, 3)
    auto b_orig = make_filled<float>({3, 2}, dtype::float32, {10, 20, 30, 40, 50, 60});
    auto b = b_orig.permute({1, 0});
    EXPECT_FALSE(b.is_contiguous());
    auto c = add(a, b);
    // b after permute is logically:
    //   10 30 50
    //   20 40 60
    // c = a + b:
    //   11 32 53
    //   24 45 66
    EXPECT_EQ(read_all<float>(c), (std::vector<float>{11.0f, 32.0f, 53.0f, 24.0f, 45.0f, 66.0f}));
}
