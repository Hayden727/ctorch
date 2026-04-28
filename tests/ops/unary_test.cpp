//===- tests/ops/unary_test.cpp - Element-wise unary CPU ops ---*- C++ -*-===//
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

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using ctorch::abs;
using ctorch::Device;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::exp;
using ctorch::log;
using ctorch::neg;
using ctorch::relu;
using ctorch::sigmoid;
using ctorch::sqrt;
using ctorch::tanh;
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

template <class T>
void expect_near(const std::vector<T>& got, const std::vector<T>& expected, T tol) {
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tol) << "i=" << i;
    }
}

} // namespace

TEST(UnaryNeg, FloatAndInt) {
    auto a = make_filled<float>({3}, dtype::float32, {1.0f, -2.0f, 3.5f});
    EXPECT_EQ(read_all<float>(neg(a)), (std::vector<float>{-1.0f, 2.0f, -3.5f}));

    auto b = make_filled<std::int64_t>({4}, dtype::int64, {1, -2, 3, 0});
    EXPECT_EQ(read_all<std::int64_t>(neg(b)), (std::vector<std::int64_t>{-1, 2, -3, 0}));
}

TEST(UnaryAbs, FloatAndInt) {
    auto a = make_filled<float>({3}, dtype::float32, {1.0f, -2.0f, -3.5f});
    EXPECT_EQ(read_all<float>(abs(a)), (std::vector<float>{1.0f, 2.0f, 3.5f}));

    auto b = make_filled<std::int32_t>({3}, dtype::int32, {-7, 0, 7});
    EXPECT_EQ(read_all<std::int32_t>(abs(b)), (std::vector<std::int32_t>{7, 0, 7}));
}

TEST(UnaryRelu, ClampsNegativeToZero) {
    auto a = make_filled<float>({4}, dtype::float32, {-1.0f, 0.0f, 0.5f, 2.0f});
    EXPECT_EQ(read_all<float>(relu(a)), (std::vector<float>{0.0f, 0.0f, 0.5f, 2.0f}));
}

TEST(UnaryExp, KnownValues) {
    auto a = make_filled<float>({3}, dtype::float32, {0.0f, 1.0f, 2.0f});
    expect_near<float>(read_all<float>(exp(a)), {1.0f, std::exp(1.0f), std::exp(2.0f)}, 1e-5f);
}

TEST(UnaryLog, KnownValues) {
    auto a = make_filled<double>({2}, dtype::float64, {1.0, std::exp(1.0)});
    expect_near<double>(read_all<double>(log(a)), {0.0, 1.0}, 1e-12);
}

TEST(UnarySqrt, KnownValues) {
    auto a = make_filled<float>({4}, dtype::float32, {0.0f, 1.0f, 4.0f, 9.0f});
    expect_near<float>(read_all<float>(sqrt(a)), {0.0f, 1.0f, 2.0f, 3.0f}, 1e-5f);
}

TEST(UnarySigmoid, MidpointAndAsymptotes) {
    auto a = make_filled<float>({3}, dtype::float32, {0.0f, -10.0f, 10.0f});
    auto out = read_all<float>(sigmoid(a));
    EXPECT_NEAR(out[0], 0.5f, 1e-6f);
    EXPECT_LT(out[1], 1e-3f);
    EXPECT_GT(out[2], 1.0f - 1e-3f);
}

TEST(UnaryTanh, MidpointAndAsymptotes) {
    auto a = make_filled<float>({3}, dtype::float32, {0.0f, -5.0f, 5.0f});
    auto out = read_all<float>(tanh(a));
    EXPECT_NEAR(out[0], 0.0f, 1e-6f);
    EXPECT_NEAR(out[1], -1.0f, 1e-3f);
    EXPECT_NEAR(out[2], 1.0f, 1e-3f);
}

TEST(UnaryNeg, OperatorMinusOnTensor) {
    auto a = make_filled<float>({2}, dtype::float32, {3.0f, -4.0f});
    auto b = -a;
    EXPECT_EQ(read_all<float>(b), (std::vector<float>{-3.0f, 4.0f}));
}

TEST(UnaryExp, RejectsIntInput) {
    auto a = make_filled<std::int32_t>({2}, dtype::int32, {1, 2});
    EXPECT_THROW(exp(a), DTypeError);
}

TEST(UnaryNeg, RejectsBoolInput) {
    auto a = make_filled<bool>({2}, dtype::bool_, {true, false});
    EXPECT_THROW(neg(a), DTypeError);
}

TEST(UnaryRelu, PropagatesNaN) {
    // PyTorch propagates NaN through relu; the naive `a > 0 ? a : 0`
    // ternary silently flips NaN to 0 because `NaN > 0` is false.
    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    auto a = make_filled<float>({3}, dtype::float32, {kNaN, -1.0f, 2.0f});
    auto out = read_all<float>(relu(a));
    EXPECT_TRUE(std::isnan(out[0])) << "got " << out[0];
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
}

TEST(UnaryAbs, NegativeZeroBecomesPositiveZero) {
    // abs(-0.0) should clear the sign bit, matching IEEE 754 / fabs.
    auto a = make_filled<float>({1}, dtype::float32, {-0.0f});
    const float got = read_all<float>(abs(a))[0];
    EXPECT_EQ(got, 0.0f);
    EXPECT_FALSE(std::signbit(got)) << "abs(-0.0) returned -0.0";
}

TEST(UnaryAbs, FloatPreservesNaN) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    auto a = make_filled<double>({1}, dtype::float64, {kNaN});
    EXPECT_TRUE(std::isnan(read_all<double>(abs(a))[0]));
}

TEST(UnaryNeg, IntMinDoesNotOverflow) {
    // Signed-min negation is UB by the standard; ours uses unsigned arithmetic
    // so the bit pattern wraps cleanly to itself.
    constexpr auto kInt32Min = std::numeric_limits<std::int32_t>::min();
    constexpr auto kInt64Min = std::numeric_limits<std::int64_t>::min();
    auto a = make_filled<std::int32_t>({1}, dtype::int32, {kInt32Min});
    auto b = make_filled<std::int64_t>({1}, dtype::int64, {kInt64Min});
    EXPECT_EQ(read_all<std::int32_t>(neg(a))[0], kInt32Min);
    EXPECT_EQ(read_all<std::int64_t>(neg(b))[0], kInt64Min);
    // abs(INT_MIN) likewise wraps to INT_MIN rather than triggering UB.
    EXPECT_EQ(read_all<std::int32_t>(abs(a))[0], kInt32Min);
    EXPECT_EQ(read_all<std::int64_t>(abs(b))[0], kInt64Min);
}

TEST(UnaryAbs, NonContiguousInputViaPermute) {
    // a is (2,3) contiguous; permute to (3,2) — non-contiguous strides.
    auto a = make_filled<float>({2, 3}, dtype::float32, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    auto a_t = a.permute({1, 0});
    EXPECT_FALSE(a_t.is_contiguous());
    auto out = abs(a_t);
    EXPECT_EQ(out.shape(), std::vector<std::int64_t>({3, 2}));
    // a_t logical layout (row-major):
    //   1 -4
    //  -2  5
    //   3 -6
    // After abs (writing into a fresh contiguous buffer):
    //   1 4
    //   2 5
    //   3 6
    EXPECT_EQ(read_all<float>(out), (std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
}
