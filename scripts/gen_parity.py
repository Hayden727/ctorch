#!/usr/bin/env python3
"""Regenerate the .npy fixtures consumed by tests/parity/.

Run this script whenever the fixture catalog or its inputs change, then
commit the updated .npy files in tests/parity/fixtures/. The C++ test
suite is intentionally decoupled from any Python runtime — CI loads the
committed fixtures directly via tests/parity/load_npy.{h,cpp}, so PyTorch
is not required in CI even though it (or numpy) is required to run this
script locally.

Each binary fixture produces three files:
    <op>_<dtype>_<shape_a>_<shape_b>_a.npy
    <op>_<dtype>_<shape_a>_<shape_b>_b.npy
    <op>_<dtype>_<shape_a>_<shape_b>_ref.npy

Each unary fixture produces two:
    <op>_<dtype>_<shape>_in.npy
    <op>_<dtype>_<shape>_ref.npy

The tolerance per (op, dtype) lives in the C++ test, not here — see
tests/parity/binary_parity_test.cpp / unary_parity_test.cpp.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np

OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "tests", "parity", "fixtures")
)

RNG = np.random.default_rng(seed=0)


def shape_tag(shape: Iterable[int]) -> str:
    return "x".join(str(d) for d in shape) if list(shape) else "scalar"


def random_input(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        return RNG.integers(-32, 32, size=shape, dtype=dtype)
    return RNG.standard_normal(size=shape).astype(dtype)


def positive_input(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Strictly-positive inputs for log/sqrt."""
    return (RNG.standard_normal(size=shape) ** 2 + 0.1).astype(dtype)


# ---- catalogs -----------------------------------------------------------

BINARY_CASES = [
    ("add", np.float32, (3, 4), (3, 4)),
    ("add", np.float32, (3, 1), (1, 4)),    # broadcast
    ("add", np.float64, (5,), (5,)),
    ("add", np.int32, (4,), (4,)),
    ("add", np.int64, (2, 3), (3,)),         # broadcast across leading dim
    ("sub", np.float32, (3, 4), (3, 4)),
    ("sub", np.float64, (2, 2), (2, 2)),
    ("sub", np.int32, (5,), (5,)),
    ("mul", np.float32, (3, 1), (1, 4)),    # outer product
    ("mul", np.float64, (4,), (4,)),
    ("mul", np.int32, (3, 3), (3, 3)),
    ("div", np.float32, (4, 4), (4, 4)),
    ("div", np.float64, (3,), (3,)),
]

# Unary cases: (op_name, dtype, shape, generator)
#   generator chooses between random_input (default) and positive_input.
UNARY_CASES = [
    ("neg", np.float32, (3, 4), random_input),
    ("neg", np.int32, (5,), random_input),
    ("abs", np.float32, (3, 4), random_input),
    ("abs", np.int64, (5,), random_input),
    ("relu", np.float32, (8,), random_input),
    ("exp", np.float32, (4,), random_input),
    ("exp", np.float64, (4,), random_input),
    ("log", np.float32, (4,), positive_input),
    ("log", np.float64, (4,), positive_input),
    ("sqrt", np.float32, (4,), positive_input),
    ("sqrt", np.float64, (4,), positive_input),
    ("sigmoid", np.float32, (8,), random_input),
    ("sigmoid", np.float64, (8,), random_input),
    ("tanh", np.float32, (8,), random_input),
    ("tanh", np.float64, (8,), random_input),
]

BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}

UNARY_OPS = {
    "neg": lambda x: -x,
    "abs": lambda x: np.abs(x),
    "relu": lambda x: np.maximum(x, 0).astype(x.dtype),
    "exp": lambda x: np.exp(x).astype(x.dtype),
    "log": lambda x: np.log(x).astype(x.dtype),
    "sqrt": lambda x: np.sqrt(x).astype(x.dtype),
    "sigmoid": lambda x: (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(x.dtype),
    "tanh": lambda x: np.tanh(x).astype(x.dtype),
}


# ---- reduction catalogs ----------------------------------------------

# (op_name, dtype, shape, dims, keepdim) — `dims=None` means whole-tensor.
SUM_LIKE_CASES = [
    ("sum", np.float32, (3, 4), None, False),
    ("sum", np.float32, (3, 4), (1,), False),
    ("sum", np.float32, (3, 4), (1,), True),
    ("sum", np.float32, (2, 3, 4), (0, 2), False),
    ("sum", np.float64, (8,), None, False),
    ("sum", np.int32, (4, 4), None, False),  # int -> int64
    ("sum", np.int32, (3, 4), (0,), False),
    ("mean", np.float32, (3, 4), (1,), False),
    ("mean", np.float64, (5,), None, False),
    ("mean", np.float32, (2, 3, 4), (0, 2), True),
    ("prod", np.float32, (3,), None, False),
    ("prod", np.int32, (4,), None, False),  # int -> int64
    ("prod", np.float32, (2, 3), (1,), False),
]

# Multi-axis or whole-tensor max/min, values only.
MAX_MIN_VALUE_CASES = [
    ("max", np.float32, (4,), None, False),
    ("max", np.float32, (3, 4), (1,), False),
    ("max", np.int32, (3, 4), (0,), True),
    ("min", np.float32, (3, 4), (0,), True),
    ("min", np.int64, (3, 4), None, False),
]

# Single-axis max/min returning values + indices.
MAX_MIN_IDX_CASES = [
    # (op, dtype, shape, axis, keepdim)
    ("max", np.float32, (3, 4), 1, False),
    ("min", np.int32, (3, 2), 0, False),
    ("max", np.float32, (2, 3, 4), -1, True),
]

# Single-axis argmax / argmin.
ARG_CASES = [
    ("argmax", np.float32, (3, 4), 1, False),
    ("argmin", np.int64, (5,), 0, False),
    # Tied input locks in the "first occurrence wins" rule.
    ("argmax", np.float32, (5,), 0, False, "tied"),
]

# index_select cases:
#   (src_dtype, src_shape, dim, indices, idx_dtype)
INDEX_SELECT_CASES = [
    (np.float32, (4, 3),    0, [2, 0, 3, 0],         np.int64),
    (np.float32, (3, 4),    1, [3, 1, 0],            np.int32),
    (np.float64, (5,),      0, [4, 2, 0],            np.int64),
    (np.int32,   (3, 4),    0, [1, 1, 2],            np.int64),
    (np.int64,   (2, 3, 4), 2, [3, 0],               np.int32),
    (np.float32, (2, 3, 4), 1, [2, 1, 0],            np.int64),
]


def dim_tag(dims):
    if dims is None:
        return "dimsAll"
    if isinstance(dims, int):
        sign = "n" if dims < 0 else ""
        return f"dim{sign}{abs(dims)}"
    return "dims" + "_".join(("n" + str(abs(d))) if d < 0 else str(d) for d in dims)


def kd_tag(keepdim):
    return "kd1" if keepdim else "kd0"


def reduction_input(shape, dt, suffix=None):
    if suffix == "tied":
        # Specific tied-value sequence: argmax should return index 1.
        return np.array([1.0, 3.0, 3.0, 2.0, 3.0], dtype=dt)
    return random_input(shape, dt)


def emit_sum_like(out_dir):
    written = 0
    for op_name, dt, shape, dims, keepdim in SUM_LIKE_CASES:
        x = reduction_input(shape, dt)
        if op_name == "sum":
            ref = np.sum(x, axis=dims, keepdims=keepdim)
            if np.issubdtype(dt, np.integer) or dt == np.bool_:
                ref = ref.astype(np.int64)
        elif op_name == "mean":
            # Mean only runs on float per ctorch's API.
            ref = np.mean(x, axis=dims, keepdims=keepdim).astype(dt)
        elif op_name == "prod":
            ref = np.prod(x, axis=dims, keepdims=keepdim)
            if np.issubdtype(dt, np.integer) or dt == np.bool_:
                ref = ref.astype(np.int64)
        else:
            raise ValueError(f"unknown sum-like op: {op_name}")
        prefix = f"{op_name}_{np.dtype(dt).name}_{shape_tag(shape)}_{dim_tag(dims)}_{kd_tag(keepdim)}"
        np.save(os.path.join(out_dir, prefix + "_in.npy"), x, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref, allow_pickle=False)
        written += 2
    return written


def emit_max_min_values(out_dir):
    written = 0
    for op_name, dt, shape, dims, keepdim in MAX_MIN_VALUE_CASES:
        x = reduction_input(shape, dt)
        ref = (np.max if op_name == "max" else np.min)(x, axis=dims, keepdims=keepdim)
        prefix = f"{op_name}val_{np.dtype(dt).name}_{shape_tag(shape)}_{dim_tag(dims)}_{kd_tag(keepdim)}"
        np.save(os.path.join(out_dir, prefix + "_in.npy"), x, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref, allow_pickle=False)
        written += 2
    return written


def emit_max_min_idx(out_dir):
    written = 0
    for op_name, dt, shape, axis, keepdim in MAX_MIN_IDX_CASES:
        x = reduction_input(shape, dt)
        if op_name == "max":
            ref_val = np.max(x, axis=axis, keepdims=keepdim)
            ref_idx = np.argmax(x, axis=axis, keepdims=keepdim).astype(np.int64)
        else:
            ref_val = np.min(x, axis=axis, keepdims=keepdim)
            ref_idx = np.argmin(x, axis=axis, keepdims=keepdim).astype(np.int64)
        prefix = f"{op_name}idx_{np.dtype(dt).name}_{shape_tag(shape)}_{dim_tag(axis)}_{kd_tag(keepdim)}"
        np.save(os.path.join(out_dir, prefix + "_in.npy"), x, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref_val, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref_idx.npy"), ref_idx, allow_pickle=False)
        written += 3
    return written


def emit_arg(out_dir):
    written = 0
    for case in ARG_CASES:
        op_name, dt, shape, axis, keepdim = case[:5]
        suffix = case[5] if len(case) > 5 else None
        x = reduction_input(shape, dt, suffix=suffix)
        if op_name == "argmax":
            ref = np.argmax(x, axis=axis, keepdims=keepdim).astype(np.int64)
        else:
            ref = np.argmin(x, axis=axis, keepdims=keepdim).astype(np.int64)
        suffix_tag = f"_{suffix}" if suffix is not None else ""
        prefix = (
            f"{op_name}_{np.dtype(dt).name}_{shape_tag(shape)}_{dim_tag(axis)}_"
            f"{kd_tag(keepdim)}{suffix_tag}"
        )
        np.save(os.path.join(out_dir, prefix + "_in.npy"), x, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref, allow_pickle=False)
        written += 2
    return written


def emit_binary(out_dir: str) -> int:
    written = 0
    for op_name, dt, shape_a, shape_b in BINARY_CASES:
        a = random_input(shape_a, dt)
        b = random_input(shape_b, dt)
        # Avoid divide-by-zero in float division fixtures.
        if op_name == "div":
            b = b + np.where(np.abs(b) < 0.1, 0.5, 0.0).astype(dt)
        ref = BINARY_OPS[op_name](a, b)
        prefix = f"{op_name}_{np.dtype(dt).name}_{shape_tag(shape_a)}_{shape_tag(shape_b)}"
        np.save(os.path.join(out_dir, prefix + "_a.npy"), a, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_b.npy"), b, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref, allow_pickle=False)
        written += 3
    return written


def emit_index_select(out_dir: str) -> int:
    written = 0
    for src_dt, src_shape, dim, indices, idx_dt in INDEX_SELECT_CASES:
        x = random_input(src_shape, src_dt)
        idx = np.asarray(indices, dtype=idx_dt)
        # np.take with axis= matches PyTorch / ctorch index_select semantics.
        ref = np.take(x, idx, axis=dim)
        prefix = (
            f"index_select_{np.dtype(src_dt).name}_{shape_tag(src_shape)}_"
            f"dim{dim}_{np.dtype(idx_dt).name}_n{len(indices)}"
        )
        np.save(os.path.join(out_dir, prefix + "_in.npy"), x, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_idx.npy"), idx, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref, allow_pickle=False)
        written += 3
    return written


def emit_unary(out_dir: str) -> int:
    written = 0
    for op_name, dt, shape, gen in UNARY_CASES:
        x = gen(shape, dt)
        ref = UNARY_OPS[op_name](x)
        prefix = f"{op_name}_{np.dtype(dt).name}_{shape_tag(shape)}"
        np.save(os.path.join(out_dir, prefix + "_in.npy"), x, allow_pickle=False)
        np.save(os.path.join(out_dir, prefix + "_ref.npy"), ref, allow_pickle=False)
        written += 2
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=OUT_DIR, help="output directory (default: tests/parity/fixtures)"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_bin = emit_binary(args.out)
    n_un = emit_unary(args.out)
    n_sum_like = emit_sum_like(args.out)
    n_mm_val = emit_max_min_values(args.out)
    n_mm_idx = emit_max_min_idx(args.out)
    n_arg = emit_arg(args.out)
    n_idx = emit_index_select(args.out)
    total = n_bin + n_un + n_sum_like + n_mm_val + n_mm_idx + n_arg + n_idx
    print(f"wrote {total} files to {args.out}")


if __name__ == "__main__":
    main()
