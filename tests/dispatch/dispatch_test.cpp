//===- tests/dispatch/dispatch_test.cpp - Dispatch table ------------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies §AC7: registering a dummy op and calling it through the dispatch
/// API returns the registered function. No real ops are defined yet.
///
//===----------------------------------------------------------------------===//

#include "ctorch/device.h"
#include "ctorch/dispatch.h"

#include <gtest/gtest.h>

namespace {

struct DummyOp {
    using fn_t = int (*)(int);
};

int cpu_impl(int x) { return x + 1; }
int cuda_impl(int x) { return x * 10; }

} // namespace

TEST(Dispatch, RegisterAndCallReturnsRegisteredFunction) {
    ctorch::dispatch::register_op<DummyOp>(ctorch::Device::Kind::CPU, &cpu_impl);
    EXPECT_TRUE(ctorch::dispatch::has_op<DummyOp>(ctorch::Device::Kind::CPU));
    EXPECT_EQ(ctorch::dispatch::call<DummyOp>(ctorch::Device::Kind::CPU, 41), 42);
}

TEST(Dispatch, PerDeviceTablesAreIndependent) {
    ctorch::dispatch::register_op<DummyOp>(ctorch::Device::Kind::CPU, &cpu_impl);
    ctorch::dispatch::register_op<DummyOp>(ctorch::Device::Kind::CUDA, &cuda_impl);

    EXPECT_EQ(ctorch::dispatch::call<DummyOp>(ctorch::Device::Kind::CPU, 5), 6);
    EXPECT_EQ(ctorch::dispatch::call<DummyOp>(ctorch::Device::Kind::CUDA, 5), 50);
}

TEST(Dispatch, CallWithoutRegistrationThrows) {
    struct UnregisteredOp {
        using fn_t = int (*)(int);
    };
    EXPECT_FALSE(ctorch::dispatch::has_op<UnregisteredOp>(ctorch::Device::Kind::CPU));
    EXPECT_THROW((void)ctorch::dispatch::call<UnregisteredOp>(ctorch::Device::Kind::CPU, 0),
                 std::runtime_error);
}
