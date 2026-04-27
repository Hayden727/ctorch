//===- tests/smoke/smoke_test.cpp - GoogleTest plumbing smoke test --------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke test that validates the GoogleTest + CTest plumbing.
///
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

TEST(Smoke, BuildSucceeds) { EXPECT_EQ(2 + 2, 4); }
