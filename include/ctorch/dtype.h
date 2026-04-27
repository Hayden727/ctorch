//===- include/ctorch/dtype.h - Tensor element data types ------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Element data type tag and per-element byte sizes.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_DTYPE_H
#define CTORCH_DTYPE_H

#include <cstddef>
#include <stdexcept>

namespace ctorch {

/// Element data type. `bfloat16` is a tag only at this stage; arithmetic on it
/// is introduced together with operators (Issue 03+).
enum class dtype : unsigned char {
    float32,
    float64,
    int32,
    int64,
    bool_,
    bfloat16,
};

/// Size in bytes of a single element of \p dt.
constexpr std::size_t size_of(dtype dt) {
    switch (dt) {
    case dtype::float32:
        return 4;
    case dtype::float64:
        return 8;
    case dtype::int32:
        return 4;
    case dtype::int64:
        return 8;
    case dtype::bool_:
        return 1;
    case dtype::bfloat16:
        return 2;
    }
    throw std::invalid_argument("ctorch::size_of: unknown dtype");
}

} // namespace ctorch

#endif // CTORCH_DTYPE_H
