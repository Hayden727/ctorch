//===- tests/ops/promotion_test.cpp - dtype promotion ----------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/type_promotion.h"

#include <gtest/gtest.h>

using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::promote_types;

TEST(Promotion, Identity) {
    EXPECT_EQ(promote_types(dtype::bool_, dtype::bool_), dtype::bool_);
    EXPECT_EQ(promote_types(dtype::int32, dtype::int32), dtype::int32);
    EXPECT_EQ(promote_types(dtype::int64, dtype::int64), dtype::int64);
    EXPECT_EQ(promote_types(dtype::float32, dtype::float32), dtype::float32);
    EXPECT_EQ(promote_types(dtype::float64, dtype::float64), dtype::float64);
}

TEST(Promotion, Symmetric) {
    constexpr dtype kAll[] = {dtype::bool_, dtype::int32, dtype::int64, dtype::float32,
                              dtype::float64};
    for (dtype a : kAll) {
        for (dtype b : kAll) {
            EXPECT_EQ(promote_types(a, b), promote_types(b, a)) << "asymmetry";
        }
    }
}

TEST(Promotion, BoolWidens) {
    EXPECT_EQ(promote_types(dtype::bool_, dtype::int32), dtype::int32);
    EXPECT_EQ(promote_types(dtype::bool_, dtype::int64), dtype::int64);
    EXPECT_EQ(promote_types(dtype::bool_, dtype::float32), dtype::float32);
    EXPECT_EQ(promote_types(dtype::bool_, dtype::float64), dtype::float64);
}

TEST(Promotion, IntegerLattice) {
    EXPECT_EQ(promote_types(dtype::int32, dtype::int64), dtype::int64);
    EXPECT_EQ(promote_types(dtype::int32, dtype::float32), dtype::float32);
    EXPECT_EQ(promote_types(dtype::int32, dtype::float64), dtype::float64);
    EXPECT_EQ(promote_types(dtype::int64, dtype::float32), dtype::float32);
    EXPECT_EQ(promote_types(dtype::int64, dtype::float64), dtype::float64);
}

TEST(Promotion, FloatLattice) {
    EXPECT_EQ(promote_types(dtype::float32, dtype::float64), dtype::float64);
    EXPECT_EQ(promote_types(dtype::float64, dtype::float32), dtype::float64);
}

TEST(Promotion, BFloat16Throws) {
    EXPECT_THROW(promote_types(dtype::bfloat16, dtype::float32), DTypeError);
    EXPECT_THROW(promote_types(dtype::float32, dtype::bfloat16), DTypeError);
    EXPECT_THROW(promote_types(dtype::bfloat16, dtype::bfloat16), DTypeError);
}
