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
    print(f"wrote {n_bin + n_un} files to {args.out}")


if __name__ == "__main__":
    main()
