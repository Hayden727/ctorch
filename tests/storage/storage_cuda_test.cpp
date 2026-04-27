//===- tests/storage/storage_cuda_test.cpp - Cross-device init -----------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies that constructing a CUDA Storage on a device that is not the
/// caller's current device still zero-initializes the buffer correctly and
/// leaves the caller's current device unchanged. Regression test for the
/// case where `cudaMemset` ran without a device guard and consequently
/// targeted the wrong CUDA context.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/storage.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>

TEST(StorageCuda, ZeroInitWorksOnNonCurrentCudaDevice) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "needs at least two visible CUDA devices";
    }

    // Pin the caller thread to device 0, then construct storage on device 1.
    // Without a device guard around cudaMemset, the zero-fill would target
    // device 0's context against a device-1 pointer.
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr std::size_t kBytes = 256;
    ctorch::Storage s(kBytes, ctorch::Device::cuda(1));
    ASSERT_TRUE(s.defined());

    // Caller's current device must be untouched by the construction.
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 0);

    // Read the buffer back and confirm every byte is 0.
    std::array<unsigned char, kBytes> readback{};
    readback.fill(0xAB);
    ASSERT_EQ(cudaMemcpy(readback.data(), s.data(), kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::size_t i = 0; i < kBytes; ++i) {
        EXPECT_EQ(readback[i], 0u) << "byte " << i;
    }
}

TEST(StorageCuda, ConstructionPreservesCallerCurrentDevice) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
        GTEST_SKIP() << "needs at least one visible CUDA device";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    {
        ctorch::Storage s(64, ctorch::Device::cuda(0));
        ASSERT_TRUE(s.defined());
    }

    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 0);
}
