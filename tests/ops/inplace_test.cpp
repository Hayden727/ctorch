//===- tests/ops/inplace_test.cpp - In-place binary CPU ops ----*- C++ -*-===//
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

using ctorch::add_;
using ctorch::AliasError;
using ctorch::Device;
using ctorch::div_;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::mul_;
using ctorch::ShapeError;
using ctorch::sub_;
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

TEST(InplaceAdd, SameDtypeFloat) {
    auto a = make_filled<float>({3}, dtype::float32, {1.0f, 2.0f, 3.0f});
    auto b = make_filled<float>({3}, dtype::float32, {10.0f, 20.0f, 30.0f});
    add_(a, b);
    EXPECT_EQ(read_all<float>(a), (std::vector<float>{11.0f, 22.0f, 33.0f}));
}

TEST(InplaceMul, BroadcastsRhs) {
    auto a = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    auto b = make_filled<float>({3}, dtype::float32, {10, 100, 1000});
    mul_(a, b);
    EXPECT_EQ(read_all<float>(a),
              (std::vector<float>{10, 200, 3000, 40, 500, 6000}));
}

TEST(InplaceSub, RejectsShapeChange) {
    // out shape (2,3) cannot equal broadcast of (2,3) and (4,) — mismatch in
    // the last dim — so add_ must throw ShapeError, not silently reshape.
    auto a = make_filled<float>({2, 3}, dtype::float32, {1, 2, 3, 4, 5, 6});
    auto b = make_filled<float>({4}, dtype::float32, {1, 2, 3, 4});
    EXPECT_THROW(sub_(a, b), ShapeError);
}

TEST(InplaceAdd, RejectsDtypeChange) {
    // Promoted dtype is float32 but a is int32 — would require widening a.
    auto a = make_filled<std::int32_t>({2}, dtype::int32, {1, 2});
    auto b = make_filled<float>({2}, dtype::float32, {0.5f, 0.5f});
    EXPECT_THROW(add_(a, b), DTypeError);
}

TEST(InplaceAdd, AcceptsExactSelfAlias) {
    // a += a is allowed: a aliases the rhs but it's the *same* view.
    auto a = make_filled<float>({3}, dtype::float32, {1.0f, 2.0f, 3.0f});
    add_(a, a);
    EXPECT_EQ(read_all<float>(a), (std::vector<float>{2.0f, 4.0f, 6.0f}));
}

TEST(InplaceAdd, RejectsNonTrivialAliasViaPermute) {
    auto a = make_filled<float>({3, 3}, dtype::float32,
                                {1, 2, 3, 4, 5, 6, 7, 8, 9});
    // Permute creates a same-storage view with different stride pattern —
    // overlapping byte ranges but not a same-view operand.
    auto a_t = a.permute({1, 0});
    EXPECT_THROW(add_(a, a_t), AliasError);
}

TEST(InplaceDiv, FloatDivisionWritesIntoLhs) {
    auto a = make_filled<float>({3}, dtype::float32, {6.0f, 9.0f, 12.0f});
    auto b = make_filled<float>({3}, dtype::float32, {2.0f, 3.0f, 4.0f});
    div_(a, b);
    EXPECT_EQ(read_all<float>(a), (std::vector<float>{3.0f, 3.0f, 3.0f}));
}
