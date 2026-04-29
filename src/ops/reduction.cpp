//===- src/ops/reduction.cpp - Reduction front-doors + helpers ------------===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Pure-CPU helpers — axis canonicalisation, output-shape arithmetic,
/// dtype-rule guards — and the public front-door free functions for
/// every reduction op declared in `<ctorch/ops/reduction.h>`. The
/// front-door allocates the output tensor(s) and dispatches to the
/// per-device kernel registered against the matching OpKey.
///
//===----------------------------------------------------------------------===//

#include "ctorch/ops/reduction.h"

#include "ctorch/device.h"
#include "ctorch/dispatch.h"
#include "ctorch/dtype.h"
#include "ctorch/errors.h"
#include "ctorch/ops/op_keys.h"
#include "ctorch/tensor.h"

#include "ops/reduction.h"
#include "ops/tensor_iter.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ctorch::ops {

namespace {

[[nodiscard]] std::int64_t normalise_axis(std::int64_t dim, int rank, const char* name) {
    const std::int64_t r = rank;
    const std::int64_t adjusted = dim < 0 ? dim + r : dim;
    if (adjusted < 0 || adjusted >= r) {
        throw ShapeError(std::string("ctorch::") + name + ": axis " + std::to_string(dim) +
                         " out of range for tensor of rank " + std::to_string(rank));
    }
    return adjusted;
}

} // namespace

ReductionAxes canonicalise(const Tensor& x, std::vector<std::int64_t> dims) {
    const auto& shape = x.shape();
    if (shape.size() > kMaxRank) {
        throw ShapeError("ctorch: tensor rank exceeds kMaxRank");
    }
    ReductionAxes ax;
    ax.rank = static_cast<int>(shape.size());

    if (dims.empty()) {
        // Whole-tensor reduction: collapse every axis.
        for (int d = 0; d < ax.rank; ++d) {
            ax.reduce[static_cast<std::size_t>(d)] = true;
        }
    } else {
        for (std::int64_t raw : dims) {
            const std::int64_t d = normalise_axis(raw, ax.rank, "reduction");
            auto& slot = ax.reduce[static_cast<std::size_t>(d)];
            if (slot) {
                throw ShapeError("ctorch: duplicate axis " + std::to_string(d) +
                                 " in reduction dims");
            }
            slot = true;
        }
    }

    ax.reduced_numel = 1;
    ax.kept_numel = 1;
    for (int d = 0; d < ax.rank; ++d) {
        const std::int64_t size = shape[static_cast<std::size_t>(d)];
        if (ax.reduce[static_cast<std::size_t>(d)]) {
            ax.reduced_numel *= size;
        } else {
            ax.kept_numel *= size;
        }
    }
    return ax;
}

int canonicalise_single(const Tensor& x, std::int64_t dim) {
    const int rank = static_cast<int>(x.shape().size());
    if (rank == 0) {
        throw ShapeError("ctorch: cannot reduce a 0-d tensor along a single axis "
                         "(use the whole-tensor form instead)");
    }
    return static_cast<int>(normalise_axis(dim, rank, "reduction"));
}

std::vector<std::int64_t> reduced_shape(const Tensor& x, const ReductionAxes& ax, bool keepdim) {
    const auto& in_shape = x.shape();
    std::vector<std::int64_t> out;
    out.reserve(in_shape.size());
    for (int d = 0; d < ax.rank; ++d) {
        if (ax.reduce[static_cast<std::size_t>(d)]) {
            if (keepdim) {
                out.push_back(1);
            }
        } else {
            out.push_back(in_shape[static_cast<std::size_t>(d)]);
        }
    }
    return out;
}

std::vector<std::int64_t> reduced_shape_single(const Tensor& x, int axis, bool keepdim) {
    const auto& in_shape = x.shape();
    std::vector<std::int64_t> out;
    out.reserve(in_shape.size());
    for (int d = 0; d < static_cast<int>(in_shape.size()); ++d) {
        if (d == axis) {
            if (keepdim) {
                out.push_back(1);
            }
        } else {
            out.push_back(in_shape[static_cast<std::size_t>(d)]);
        }
    }
    return out;
}

dtype reduce_sum_prod_dtype(dtype in) {
    switch (in) {
    case dtype::bool_:
    case dtype::int32:
    case dtype::int64:
        return dtype::int64;
    case dtype::float32:
    case dtype::float64:
        return in;
    case dtype::bfloat16:
        throw DTypeError("ctorch: bfloat16 reductions are not supported");
    }
    throw DTypeError("ctorch: unknown dtype in reduction");
}

void require_float_for_mean(dtype in, const char* name) {
    if (in == dtype::float32 || in == dtype::float64) {
        return;
    }
    throw DTypeError(std::string("ctorch::") + name +
                     ": requires a floating dtype input "
                     "(cast int/bool to float first)");
}

void reject_bfloat16(dtype in, const char* name) {
    if (in == dtype::bfloat16) {
        throw DTypeError(std::string("ctorch::") + name + ": bfloat16 is not supported");
    }
}

} // namespace ctorch::ops
