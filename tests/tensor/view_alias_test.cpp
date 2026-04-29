//===- tests/tensor/view_alias_test.cpp - View alias semantics ------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Round-trip tests proving that mutations through the source `Tensor`
/// are visible through views produced by `slice` / `select` / `narrow`,
/// and vice-versa.
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

float* fdata(Tensor& t) { return static_cast<float*>(t.storage().data()) + t.offset(); }

} // namespace

TEST(TensorViewAlias, SliceObservesWritesThroughSource) {
    Tensor t({4, 3}, dtype::float32, Device::cpu());
    Tensor s = t.slice(0, 1, 3); // {2,3} view of rows 1..2

    // Source-side write at row 2, col 1 → stride[0]=3, so offset = 2*3+1 = 7.
    auto* base = static_cast<float*>(t.storage().data());
    base[7] = 42.0f;

    // View row 1, col 1 == source row 2, col 1.
    const auto* sp = fdata(s);
    EXPECT_EQ(sp[1 * 3 + 1], 42.0f);
}

TEST(TensorViewAlias, SourceObservesWritesThroughView) {
    Tensor t({4, 3}, dtype::float32, Device::cpu());
    Tensor v = t.select(0, 2); // 1-D view at row 2

    auto* vp = fdata(v);
    vp[0] = 7.0f;
    vp[2] = 9.0f;

    const auto* base = static_cast<const float*>(t.storage().data());
    EXPECT_EQ(base[2 * 3 + 0], 7.0f);
    EXPECT_EQ(base[2 * 3 + 2], 9.0f);
}

TEST(TensorViewAlias, ChainedViewsAllShareStorage) {
    Tensor t({4, 4}, dtype::float32, Device::cpu());
    Tensor a = t.narrow(0, 1, 2); // {2, 4} sub-rows
    Tensor b = a.select(0, 1);    // 1-D row inside that

    EXPECT_EQ(t.storage().data(), a.storage().data());
    EXPECT_EQ(a.storage().data(), b.storage().data());

    // b is row 2 of t (rows 1..2 → second one is t row 2).
    auto* bp = fdata(b);
    bp[2] = 11.0f;

    const auto* base = static_cast<const float*>(t.storage().data());
    EXPECT_EQ(base[2 * 4 + 2], 11.0f);
}

TEST(TensorViewAlias, ViewKeepsStorageAliveAfterSourceDrop) {
    Tensor v;
    {
        Tensor t({4}, dtype::float32, Device::cpu());
        auto* p = static_cast<float*>(t.storage().data());
        p[1] = 17.0f;
        v = t.slice(0, 0, 4);
        // `t` goes out of scope here; the storage stays alive through `v`.
    }
    const auto* p = fdata(v);
    EXPECT_EQ(p[1], 17.0f);
    EXPECT_EQ(v.storage().use_count(), 1);
}
