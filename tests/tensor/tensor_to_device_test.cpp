//===- tests/tensor/tensor_to_device_test.cpp - Tensor::to(device) --------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU↔CPU `to()` covers the same-device shortcut. The CUDA round-trip
/// exercise from §AC2 is deferred to a CUDA-equipped runner; the test source
/// is wired up so it runs whenever the build sees a real CUDA backend.
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

TEST(TensorTo, SameDeviceReturnsAlias) {
    Tensor t({4}, dtype::float32, Device::cpu());
    auto* p = static_cast<float*>(t.storage().data());
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;

    Tensor u = t.to(Device::cpu());
    EXPECT_EQ(t.storage().data(), u.storage().data());
}

#if defined(CTORCH_HAS_CUDA)
TEST(TensorTo, CudaRoundTripIsByteIdentical) {
    Tensor t({2, 3}, dtype::float32, Device::cpu());
    auto* p = static_cast<float*>(t.storage().data());
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i) * 1.5f;
    }

    Tensor on_gpu = t.to(Device::cuda(0));
    Tensor back = on_gpu.to(Device::cpu());

    ASSERT_TRUE(back.device().is_cpu());
    ASSERT_EQ(back.storage().nbytes(), t.storage().nbytes());

    const auto* a = static_cast<const std::byte*>(t.storage().data());
    const auto* b = static_cast<const std::byte*>(back.storage().data());
    for (std::size_t i = 0; i < t.storage().nbytes(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "byte " << i;
    }
}
#endif
