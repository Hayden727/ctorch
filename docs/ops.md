# ctorch Element-wise Operators

Operators delivered by [Issue #3](https://github.com/Hayden727/ctorch/issues/3).
Inputs broadcast NumPy/PyTorch-style; the output dtype is the promotion of
the input dtypes; the output device matches the (single) input device.

Include `<ctorch/ops/elementwise.h>` for the free-function and operator-overload
declarations.

## Binary

| Op  | Free function       | Operator | Broadcast | Supported dtypes              | Notes                                  |
| --- | ------------------- | -------- | --------- | ----------------------------- | -------------------------------------- |
| add | `ctorch::add(a,b)`  | `a + b`  | yes       | f32, f64, i32, i64            | `bool` rejected — promote to int first |
| sub | `ctorch::sub(a,b)`  | `a - b`  | yes       | f32, f64, i32, i64            | as above                               |
| mul | `ctorch::mul(a,b)`  | `a * b`  | yes       | f32, f64, i32, i64            | as above                               |
| div | `ctorch::div(a,b)`  | `a / b`  | yes       | f32, f64                      | integer ops are rejected (use float)   |

## In-place binary

The destination is the LHS. The destination must already have the broadcast
output shape and the promoted dtype; in-place ops never reshape or rebind
storage.

| Op   | Function           | Notes                                                      |
| ---- | ------------------ | ---------------------------------------------------------- |
| add_ | `ctorch::add_(a,b)`| Throws `AliasError` if `b` is a non-trivial view of `a`.   |
| sub_ | `ctorch::sub_(a,b)`| as above                                                   |
| mul_ | `ctorch::mul_(a,b)`| as above                                                   |
| div_ | `ctorch::div_(a,b)`| as above                                                   |

## Unary

| Op      | Free function          | Operator | Supported dtypes      | Notes                                  |
| ------- | ---------------------- | -------- | --------------------- | -------------------------------------- |
| neg     | `ctorch::neg(x)`       | `-x`     | f32, f64, i32, i64    | `bool` rejected                        |
| abs     | `ctorch::abs(x)`       | —        | f32, f64, i32, i64    | `bool` rejected                        |
| relu    | `ctorch::relu(x)`      | —        | f32, f64, i32, i64    | `max(x, 0)`                            |
| exp     | `ctorch::exp(x)`       | —        | f32, f64              | int rejected                           |
| log     | `ctorch::log(x)`       | —        | f32, f64              | int rejected; positive inputs only     |
| sqrt    | `ctorch::sqrt(x)`      | —        | f32, f64              | int rejected; non-negative inputs only |
| sigmoid | `ctorch::sigmoid(x)`   | —        | f32, f64              | numerically stable two-branch impl     |
| tanh    | `ctorch::tanh(x)`      | —        | f32, f64              | uses `<cmath>` `tanhf` / `tanh`        |

## Type promotion

`ctorch::promote_types(a, b)` mirrors PyTorch's `torch.result_type` over the
five supported dtypes:

```
        bool_   int32   int64   float32  float64
bool_   bool_   int32   int64   float32  float64
int32   int32   int32   int64   float32  float64
int64   int64   int64   int64   float32  float64
float32 float32 float32 float32 float32  float64
float64 float64 float64 float64 float64  float64
```

`bfloat16` is recognised by the dtype enum but always raises `DTypeError`
in this milestone; it lands together with the mixed-precision work.

## Broadcasting

Right-aligned: the shorter shape is left-padded with leading 1s, then for
each dimension the sizes must be equal or one must be 1. Size-1 dimensions
are virtually expanded by setting that dim's stride to 0 — no copy of the
input is ever required.

## Parity tolerances (Issue 03 §N1)

| dtype   | non-transcendental | transcendental |
| ------- | ------------------ | -------------- |
| float32 | 1e-5 rel           | 1e-4 rel       |
| float64 | 1e-12 rel          | 1e-12 rel      |
| int*    | exact              | n/a            |

## Known limitations

- **`div`/`div_` reject integer operands.** Integer division by zero is UB in
  C++; rather than risk a crash on user data, ctorch refuses all integer
  division at the front door. Cast operands to `float32`/`float64` first.
  This matches PyTorch's `torch.div` default (`rounding_mode=None`) which
  also promotes integer inputs to float.
- **Non-contiguous CUDA operands cannot be implicitly promoted.** Same-dtype
  strided CUDA ops work (the strided indexer kernel handles them), but
  mixed-dtype ops on non-contiguous CUDA tensors throw `DTypeError`. A
  bespoke CUDA cast kernel is the planned fix.
- **`neg`/`abs` on `INT_MIN` wrap to `INT_MIN`.** Documented PyTorch
  behaviour. Implemented through unsigned arithmetic so it does not trip
  signed-overflow UB.

## Cross-device, dtype, and aliasing errors

| Error class    | Raised when                                                      |
| -------------- | ---------------------------------------------------------------- |
| `DeviceError`  | Operands live on different devices.                              |
| `DTypeError`   | Operand uses an unsupported dtype (`bfloat16`, bool arithmetic). |
| `ShapeError`   | Shapes cannot be broadcast / in-place result wouldn't match dst. |
| `AliasError`   | In-place destination aliases a non-identical view of the rhs.    |

All four derive from `ctorch::Error` which derives from `std::runtime_error`.

# Reductions

Operators delivered by [Issue #9](https://github.com/Hayden727/ctorch/issues/9).
Include `<ctorch/ops/reduction.h>` for the public free functions.

Negative axes are normalised against `ndim`; duplicate or out-of-range axes
raise `ShapeError`. Empty `dims` collapses every axis (whole-tensor form).
With `keepdim=true` the output keeps a singleton at every collapsed
dimension; with `keepdim=false` (default) those dimensions are dropped.

## Sum / mean / prod

| Op    | Free function                                | Notes                                                                                                                                                                                                                                                          |
| ----- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sum   | `ctorch::sum(x[, dims, keepdim])`            | bool / int* promote to int64 (matches PyTorch — int32 accumulators would overflow at 1B-element scale). fp32 sums use a `double` accumulator on CPU and CUDA so the two backends stay numerically aligned.                                                    |
| mean  | `ctorch::mean(x[, dims, keepdim])`           | Floating dtype only. Integer or `bool` input throws `DTypeError` (cast first). Empty-slice → NaN (matches PyTorch's `0/0`).                                                                                                                                    |
| prod  | `ctorch::prod(x[, dims, keepdim])`           | Same dtype rules as `sum`. Integer `prod` accumulates in int64 — saturates / wraps for very large products; cast to float for high-precision results.                                                                                                          |

## Max / min (values only)

`max(x)` / `min(x)` return a 0-d tensor of the input dtype. The
multi-axis form `max(x, dims, keepdim)` reduces every axis listed in
`dims`. Empty reductions throw `ShapeError` ("operation has no
identity"). NaN propagates: any NaN in the slice forces the result to
NaN.

## Max / min with indices (single axis)

```cpp
auto vi = ctorch::max(x, /*dim=*/-1, /*keepdim=*/false);
//   vi.values  : input dtype, shape = x.shape() with dim collapsed
//   vi.indices : int64,       shape = same as values
```

Tie-breaking is **first-occurrence-wins** (matches PyTorch's documented
behaviour and `numpy.argmax`). NaN-bearing slices produce NaN for the
value and the index of the **first** NaN.

## Argmax / argmin

`argmax(x, dim, keepdim)` and `argmin(x, dim, keepdim)` always return an
int64 index tensor. 0-d input or zero-length reduced axis throws
`ShapeError`.

## Numerical tolerances

Per-dtype parity tolerances vs PyTorch reference (verified by
`tests/parity/reduction_parity_test.cpp`):

| Input dtype | Tolerance           |
| ----------- | ------------------- |
| float32     | 1e-5 relative       |
| float64     | 1e-12 relative      |
| int*, bool  | exact               |

# Indexing & slicing

Operators delivered by [Issue #10](https://github.com/Hayden727/ctorch/issues/10).

## Zero-copy view ops

Pure metadata operations on `Tensor`. They share storage with the source
(`storage().use_count()` increments) and never launch a kernel — the
returned tensor is a strided view, not a copy.

| Op     | Method                                    | Result rank | Notes                                                                                                  |
| ------ | ----------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| slice  | `t.slice(dim, start, end, step=1)`        | unchanged   | Negative `dim`/`start`/`end` are normalised against `shape[dim]`; bounds clamp PyTorch-style. `step > 0`. |
| select | `t.select(dim, index)`                    | rank − 1    | Drops the selected dim. Negative `index` normalised; out-of-range throws `ShapeError`.                  |
| narrow | `t.narrow(dim, start, length)`            | unchanged   | Sugar for `slice(dim, start, start + length, 1)`. `length >= 0`; range must fit within `shape[dim]`.   |

`step <= 0` for `slice` raises `ShapeError`; reverse-slice is out of
scope. Empty slices (`start == end`) produce a 0-sized dim and remain
valid for downstream ops.

## index_select

`<ctorch/ops/indexing.h>`:

```cpp
Tensor index_select(const Tensor& src, int dim, const Tensor& indices);
```

Gathers along a single axis and returns a **fresh contiguous** output
(no aliasing). `indices` must be 1-D with dtype `int32` or `int64`, on
the same device as `src`. The output shape is `src.shape()` with
`src.shape[dim]` replaced by `indices.numel()`. Negative entries in
`indices` are normalised against `src.shape[dim]`; out-of-range entries
throw `ShapeError` on both backends. CUDA validates indices in a
pre-pass kernel that copies a single-int error flag back to the host.

| Source dtype          | Index dtypes  | Result dtype |
| --------------------- | ------------- | ------------ |
| f32, f64, i32, i64, bool | int32, int64 | same as source |

`bfloat16` source raises `DTypeError` (consistent with the rest of the
op surface). Non-int index dtypes raise `DTypeError`. `dim`
out-of-range, rank-0 source, or non-1-D indices raise `ShapeError`.
Cross-device pairs (`src.device() != indices.device()`) raise
`DeviceError`.

## Regenerating parity fixtures

The `.npy` fixtures under `tests/parity/fixtures/` are committed binaries.
To regenerate them after editing `scripts/gen_parity.py` (NumPy required
locally; not in CI):

```bash
python3 scripts/gen_parity.py
git add tests/parity/fixtures/
```
