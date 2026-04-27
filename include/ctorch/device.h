//===- include/ctorch/device.h - Device tag --------------------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compact device tag used by Storage, Tensor, and the dispatch table to
/// identify where a buffer lives and which backend should service an op.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_DEVICE_H
#define CTORCH_DEVICE_H

namespace ctorch {

/// POD device tag. Total size is 8 bytes (4 enum + 4 index), well within the
/// 128-byte Tensor metadata budget required by Issue 02 §N2.
struct Device {
    enum class Kind : int { CPU = 0, CUDA = 1 };

    Kind kind = Kind::CPU;
    int index = 0; ///< CUDA device ordinal; ignored on CPU.

    static constexpr Device cpu() { return Device{Kind::CPU, 0}; }
    static constexpr Device cuda(int i = 0) { return Device{Kind::CUDA, i}; }

    constexpr bool is_cpu() const { return kind == Kind::CPU; }
    constexpr bool is_cuda() const { return kind == Kind::CUDA; }

    constexpr bool operator==(const Device&) const = default;
};

/// Number of distinct device kinds. Used to size the dispatch table.
inline constexpr int kNumDeviceKinds = 2;

} // namespace ctorch

#endif // CTORCH_DEVICE_H
