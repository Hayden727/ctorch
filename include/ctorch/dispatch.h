//===- include/ctorch/dispatch.h - Operator dispatch table -----*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Header-only registration table that future operators (Issue 03+) plug
/// into. Each operator is identified by a tag type `OpKey` that exposes
/// `using fn_t = ...;` describing its callable signature. The table holds
/// one slot per `Device::Kind`. This issue ships the plumbing only — no
/// `OpKey` types are defined here.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_DISPATCH_H
#define CTORCH_DISPATCH_H

#include "ctorch/device.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace ctorch::dispatch {

namespace detail {

template <class OpKey> struct DispatchTable {
    using fn_t = typename OpKey::fn_t;
    inline static std::array<fn_t, kNumDeviceKinds> table{};
};

} // namespace detail

/// Register a backend implementation \p fn for operator \p OpKey on \p kind.
/// Re-registering replaces the previous entry. Intended to be called from
/// static initializers in op-implementation translation units.
template <class OpKey, class Fn> void register_op(Device::Kind kind, Fn fn) {
    detail::DispatchTable<OpKey>::table[static_cast<std::size_t>(kind)] = typename OpKey::fn_t(fn);
}

/// Invoke the registered backend for \p OpKey on \p kind. Throws if no
/// implementation has been registered for that combination.
template <class OpKey, class... Args> auto call(Device::Kind kind, Args&&... args) {
    auto& slot = detail::DispatchTable<OpKey>::table[static_cast<std::size_t>(kind)];
    if (slot == nullptr) {
        throw std::runtime_error("ctorch::dispatch::call: no implementation registered");
    }
    return slot(std::forward<Args>(args)...);
}

/// Returns true iff a backend is registered for the given (OpKey, kind).
template <class OpKey> bool has_op(Device::Kind kind) {
    return detail::DispatchTable<OpKey>::table[static_cast<std::size_t>(kind)] != nullptr;
}

} // namespace ctorch::dispatch

#endif // CTORCH_DISPATCH_H
