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

## Regenerating parity fixtures

The `.npy` fixtures under `tests/parity/fixtures/` are committed binaries.
To regenerate them after editing `scripts/gen_parity.py` (NumPy required
locally; not in CI):

```bash
python3 scripts/gen_parity.py
git add tests/parity/fixtures/
```
