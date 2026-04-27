//===- src/type_promotion.cpp - PyTorch promotion table --------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "ctorch/type_promotion.h"

#include "ctorch/errors.h"

#include <array>
#include <cstddef>

namespace ctorch {

namespace {

constexpr std::size_t kIndexBool = 0;
constexpr std::size_t kIndexI32 = 1;
constexpr std::size_t kIndexI64 = 2;
constexpr std::size_t kIndexF32 = 3;
constexpr std::size_t kIndexF64 = 4;
constexpr std::size_t kSupportedCount = 5;

constexpr std::size_t dtype_to_index(dtype dt) {
    switch (dt) {
    case dtype::bool_:
        return kIndexBool;
    case dtype::int32:
        return kIndexI32;
    case dtype::int64:
        return kIndexI64;
    case dtype::float32:
        return kIndexF32;
    case dtype::float64:
        return kIndexF64;
    case dtype::bfloat16:
        break;
    }
    throw DTypeError("ctorch::promote_types: bfloat16 not supported in Issue 03");
}

constexpr dtype kPromotionTable[kSupportedCount][kSupportedCount] = {
    // a = bool_
    {dtype::bool_, dtype::int32, dtype::int64, dtype::float32, dtype::float64},
    // a = int32
    {dtype::int32, dtype::int32, dtype::int64, dtype::float32, dtype::float64},
    // a = int64
    {dtype::int64, dtype::int64, dtype::int64, dtype::float32, dtype::float64},
    // a = float32
    {dtype::float32, dtype::float32, dtype::float32, dtype::float32, dtype::float64},
    // a = float64
    {dtype::float64, dtype::float64, dtype::float64, dtype::float64, dtype::float64},
};

} // namespace

dtype promote_types(dtype a, dtype b) {
    return kPromotionTable[dtype_to_index(a)][dtype_to_index(b)];
}

} // namespace ctorch
