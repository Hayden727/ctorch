//===- src/ops/cast_cpu.cpp - CPU dtype cast -------------------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ops/cast_cpu.h"

#include "ctorch/errors.h"
#include "ops/tensor_iter.h"

#include <cstdint>

namespace ctorch::ops {

namespace {

template <class Src, class Dst> Dst static_cast_value(Src x) {
    if constexpr (std::is_same_v<Src, bool> && std::is_floating_point_v<Dst>) {
        return x ? Dst(1) : Dst(0);
    } else {
        return static_cast<Dst>(x);
    }
}

template <class Src, class Dst>
void copy_with_cast(const Tensor& in, Tensor& out) {
    const auto* in_base = static_cast<const Src*>(in.storage().data());
    auto* out_base = static_cast<Dst*>(out.storage().data());
    const auto ctx = make_unary_indexer(in, out);
    for_each_n_unary(ctx, [&](std::int64_t in_off, std::int64_t out_off) {
        out_base[out_off] = static_cast_value<Src, Dst>(in_base[in_off]);
    });
}

template <class Src> void cast_to(const Tensor& in, Tensor& out) {
    switch (out.dtype()) {
    case dtype::float32:
        copy_with_cast<Src, float>(in, out);
        break;
    case dtype::float64:
        copy_with_cast<Src, double>(in, out);
        break;
    case dtype::int32:
        copy_with_cast<Src, std::int32_t>(in, out);
        break;
    case dtype::int64:
        copy_with_cast<Src, std::int64_t>(in, out);
        break;
    case dtype::bool_:
        copy_with_cast<Src, bool>(in, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch: cast to bfloat16 not supported");
    }
}

} // namespace

Tensor cast_cpu(const Tensor& t, dtype target) {
    if (t.dtype() == target) {
        return t;
    }
    if (!t.device().is_cpu()) {
        throw DeviceError("ctorch::ops::cast_cpu: tensor not on CPU");
    }
    Tensor out(t.shape(), target, t.device());
    switch (t.dtype()) {
    case dtype::float32:
        cast_to<float>(t, out);
        break;
    case dtype::float64:
        cast_to<double>(t, out);
        break;
    case dtype::int32:
        cast_to<std::int32_t>(t, out);
        break;
    case dtype::int64:
        cast_to<std::int64_t>(t, out);
        break;
    case dtype::bool_:
        cast_to<bool>(t, out);
        break;
    case dtype::bfloat16:
        throw DTypeError("ctorch: cast from bfloat16 not supported");
    }
    return out;
}

} // namespace ctorch::ops
