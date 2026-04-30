//===- tests/ops/matmul_dtype_test.cpp - matmul dtype rules ---------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// dtype-related coverage for `ctorch::matmul`: float promotion to the
/// wider operand, integer rejection, cross-device error, undefined
/// inputs.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/linalg.h"
#include "ctorch/tensor.h"

#include <gtest/gtest.h>

using ctorch::Device;
using ctorch::dtype;
using ctorch::DTypeError;
using ctorch::matmul;
using ctorch::ShapeError;
using ctorch::Tensor;

TEST(MatmulDType, FloatPromotesToWidestFloat) {
    Tensor a({2, 3}, dtype::float32, Device::cpu());
    Tensor b({3, 2}, dtype::float64, Device::cpu());
    Tensor c = matmul(a, b);
    EXPECT_EQ(c.dtype(), dtype::float64);
}

TEST(MatmulDType, RejectsIntegerInputs) {
    Tensor a({2, 3}, dtype::int32, Device::cpu());
    Tensor b({3, 2}, dtype::int32, Device::cpu());
    EXPECT_THROW((void)matmul(a, b), DTypeError);
}

TEST(MatmulDType, RejectsBoolInputs) {
    Tensor a({2, 3}, dtype::bool_, Device::cpu());
    Tensor b({3, 2}, dtype::bool_, Device::cpu());
    EXPECT_THROW((void)matmul(a, b), DTypeError);
}

TEST(MatmulDType, RejectsBFloat16) {
    Tensor a({2, 3}, dtype::bfloat16, Device::cpu());
    Tensor b({3, 2}, dtype::bfloat16, Device::cpu());
    // promote_types throws DTypeError for bfloat16 first.
    EXPECT_THROW((void)matmul(a, b), DTypeError);
}

TEST(MatmulDType, UndefinedInputThrows) {
    Tensor a;
    Tensor b({3, 2}, dtype::float32, Device::cpu());
    EXPECT_THROW((void)matmul(a, b), ShapeError);
    EXPECT_THROW((void)matmul(b, a), ShapeError);
}
