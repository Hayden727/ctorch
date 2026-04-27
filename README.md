<!-- PROJECT LOGO -->
<div align="center">
  <img src="assets/profile.png" alt="ctorch" width="1200">
</div>

<br />

<!-- PROJECT SHIELDS -->
<div align="center">

[![C++][cpp-shield]][cpp-url]
[![CMake][cmake-shield]][cmake-url]
[![CUDA][cuda-shield]][cuda-url]
[![Claude Code][claude-shield]][claude-url]

</div>

<div align="center">

  <h3 align="center">ctorch</h3>

  <p align="center">
    A from-scratch C++20 deep learning framework with autograd, an <code>nn::Module</code> system,
    CUDA acceleration, and a Dynamo-style graph compiler.
    <br />
    <a href="docs/"><strong>Explore the docs &raquo;</strong></a>
    <br />
    <br />
    <a href="https://github.com/Hayden727/ctorch/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/Hayden727/ctorch/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#build">Build</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#tensor-basics">Tensor Basics</a></li>
        <li><a href="#autograd">Autograd</a></li>
        <li><a href="#module--training-loop">Module &amp; Training Loop</a></li>
        <li><a href="#dynamo-capture">Dynamo Capture</a></li>
      </ul>
    </li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li>
      <a href="#contributing">Contributing</a>
      <ul>
        <li><a href="#adding-a-new-op">Adding a New Op</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

<!-- ABOUT THE PROJECT -->
## About The Project

`ctorch` is a header-mostly modern C++ library that re-implements the PyTorch user
experience &mdash; `Tensor`, `autograd`, `nn::Module`, optimizers, data loaders &mdash;
with a small surface area, a transparent dispatch path, and zero external runtime
dependencies beyond a BLAS backend and the CUDA toolkit.

It exists to expose the *internals* that PyTorch hides: storage and stride machinery,
the dynamic tape, kernel dispatch, mixed-precision plumbing, and the trace-and-compile
pipeline behind `torch.compile`. The codebase is meant to be read end-to-end &mdash; an
educational, hackable foundation for systems learning, custom kernels, and embedded
inference.

### Built With

[![C++][cpp-built-shield]][cpp-url]
[![CMake][cmake-built-shield]][cmake-url]
[![OpenBLAS][openblas-shield]][openblas-url]
[![CUDA][cuda-built-shield]][cuda-url]
[![cuBLAS][cublas-shield]][cublas-url]
[![GoogleTest][gtest-shield]][gtest-url]
<!-- ROADMAP -->
## Roadmap

The project is staged into seven phases. Each phase is intended to be picked up in
order &mdash; later phases assume the contracts established by earlier ones (Tensor
&rarr; Autograd &rarr; Module &rarr; Compiler).

**Phase 0 &mdash; Foundation**
- [ ] Project skeleton: CMake (&ge; 3.20), GoogleTest harness, `clang-format`, `clang-tidy`, CI
- [ ] `Device` abstraction (`CPU`, `CUDA`) and per-device dispatch table
- [ ] `Storage` &amp; `Tensor` core: dtype, shape, stride, contiguous/view semantics
- [ ] Allocators: CPU pool allocator + CUDA caching allocator
- [ ] CUDA toolchain integration: `nvcc` driver, host/device compilation, stream wrapper

**Phase 1 &mdash; Eager Tensor Engine** (CPU + CUDA in parallel)
- [ ] Element-wise ops with broadcasting on both backends (`add`, `sub`, `mul`, `div`, `relu`, &hellip;)
- [ ] Reductions (`sum`, `mean`, `max`, `argmax`) and indexing/slicing
- [ ] Linear algebra: `matmul`/`transpose` &rarr; OpenBLAS (CPU) and cuBLAS (CUDA)
- [ ] Type promotion, dtype casting, `to(device)` host&harr;device transfer
- [ ] Numerical-parity tests: every op matches a CPU reference within tolerance

**Phase 2 &mdash; Autograd**
- [ ] Reverse-mode AD: `Tensor::requires_grad`, dynamic tape, device-agnostic
- [ ] `Function` base with `forward`/`backward` pairs registered per op
- [ ] Gradient accumulation, retain-graph, `no_grad` / `enable_grad` scope guards
- [ ] `Tensor::backward()` and `autograd::grad()` matching PyTorch semantics
- [ ] Higher-order gradients (`create_graph`)

**Phase 3 &mdash; Neural Network API**
- [ ] `nn::Module` base with parameter/buffer registration via reflection macros
- [ ] Layers: `Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`, `Embedding`, `Dropout`, `MultiheadAttention`
- [ ] Activations &amp; loss functions: `CrossEntropy`, `MSE`, `BCE`, `NLL`
- [ ] Optimizers: `SGD` (with momentum), `Adam`, `AdamW` + LR-scheduler hooks
- [ ] `Dataset` / `DataLoader` with prefetch worker pool and pinned-memory transfer

**Phase 4 &mdash; Performance &amp; Scale**
- [ ] cuDNN bindings for conv, pooling, RNN
- [ ] Mixed-precision training: `autocast` + gradient scaler on CUDA
- [ ] Hand-written CUDA kernels for hot paths (fused softmax, layernorm, attention)
- [ ] Operator fusion for elementwise chains (eager-mode, no compiler yet)
- [ ] Distributed training: NCCL all-reduce + `DistributedDataParallel`

**Phase 5 &mdash; Dynamo-style Compiler** (`ctorch::dynamo`)
- [ ] Tracing frontend: capture eager calls into a typed FX-like graph IR
- [ ] Guard system for shape/dtype specialization &amp; re-trace on guard miss
- [ ] Graph optimizer passes: DCE, CSE, constant folding, vertical &amp; horizontal fusion
- [ ] Backend codegen: C++ JIT (LLVM ORC) for CPU, NVRTC for CUDA
- [ ] `ctorch::compile(model)` matching the `torch.compile` user experience

**Phase 6 &mdash; Ecosystem &amp; Deployment**
- [ ] ONNX import / export
- [ ] Quantization: INT8 PTQ + QAT, plus a graph-mode inference runtime
- [ ] `torch.save`-compatible serialization (zip + pickle-like archive)
- [ ] Model zoo &amp; tutorials: MLP &rarr; ResNet &rarr; Transformer &rarr; nano-GPT
- [ ] Benchmarks vs. libtorch on representative workloads
<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- C++20 compiler &mdash; clang &ge; 16 or gcc &ge; 12
- CMake &ge; 3.20
- A BLAS implementation &mdash; OpenBLAS, MKL, or Apple Accelerate
- CUDA Toolkit &ge; 12 with cuBLAS and cuDNN (for the GPU backend)
- GoogleTest (vendored or system)

### Build

Clone the repository and build with both backends enabled:

```bash
git clone https://github.com/Hayden727/ctorch.git
cd ctorch
cmake -S . -B build -DCTORCH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To build without CUDA (CPU-only) on a machine without the NVIDIA toolchain:

```bash
cmake -S . -B build -DCTORCH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
<!-- USAGE -->
## Usage

The snippets below show the intended user-facing API. Items marked **(planned)** are
not yet implemented &mdash; they document the eventual surface, not the shipping one.

### Tensor Basics

```cpp
#include <ctorch/tensor.h>

int main() {
  auto a = ctorch::randn({2, 3});           // CPU by default
  auto b = ctorch::ones({2, 3}).to(ctorch::Device::CUDA);
  auto c = a.to(ctorch::Device::CUDA) + b;  // dispatches on device
  std::cout << c.cpu() << '\n';
}
```

### Autograd

```cpp
#include <ctorch/tensor.h>

auto x = ctorch::randn({4, 4}).requires_grad_(true);
auto y = (x * x).sum();                     // builds the tape
y.backward();                               // populates x.grad()
std::cout << x.grad() << '\n';
```

### Module &amp; Training Loop

```cpp
#include <ctorch/nn.h>
#include <ctorch/optim.h>

struct MLP : ctorch::nn::Module {
  ctorch::nn::Linear fc1{784, 128}, fc2{128, 10};
  ctorch::Tensor forward(ctorch::Tensor x) {
    return fc2(ctorch::relu(fc1(x)));
  }
};

MLP model;
model.to(ctorch::Device::CUDA);
ctorch::optim::Adam opt(model.parameters(), /*lr=*/1e-3);

for (auto [x, y] : loader) {
  auto logits = model.forward(x);
  auto loss   = ctorch::nn::cross_entropy(logits, y);
  opt.zero_grad();
  loss.backward();
  opt.step();
}
```

### Dynamo Capture

```cpp
// (planned) Phase 5 — trace, optimize, and codegen.
#include <ctorch/dynamo.h>

auto compiled = ctorch::compile(model);     // shape/dtype-specialized graph
auto out      = compiled(x);                // re-traces on guard miss
```
<!-- TESTING -->
## Testing

Tests are written with GoogleTest and grouped by concern:

```
tests/
├── unit/         # tensor storage, dtype, dispatch, allocators
├── autograd/     # tape construction and backward correctness
├── integration/  # module forward/backward, optimizer convergence
└── parity/       # numerical parity vs. a PyTorch reference (Python harness)
```

Run the full suite or filter by name:

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R autograd
```
<!-- REPOSITORY STRUCTURE -->
## Repository Structure

```
ctorch/
├── include/ctorch/        # Public headers
│   ├── tensor.h           # Tensor, Storage, Device, dtype
│   ├── autograd.h         # Tape, Function, no_grad guard
│   ├── nn/                # Module, layers, losses
│   ├── optim/             # SGD, Adam, AdamW, schedulers
│   ├── cuda/              # CUDA stream/event/device wrappers
│   └── dynamo/            # Graph IR, tracer, passes, codegen
├── src/                   # Implementation (.cpp)
├── kernels/               # CUDA (.cu) hand-written kernels
├── tests/                 # GoogleTest suites (unit / autograd / parity)
├── examples/              # MLP, ResNet, nano-GPT
├── benchmarks/            # vs. libtorch on representative workloads
├── docs/                  # Design notes and ADRs
└── assets/                # Logo and diagrams
```
<!-- CONTRIBUTING -->
## Contributing

1. Fork the repository and create a feature branch off `master`.
2. Install the pre-commit hooks once per clone:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
   This wires `clang-format` and the hygiene checks into every `git commit`.
3. Add tests covering the change &mdash; new ops require a parity test.
4. Before opening a PR, run the full hook suite and the clang-tidy gate that CI
   enforces:
   ```bash
   pre-commit run --all-files
   cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   clang-tidy-16 -p build $(git ls-files '*.cpp')
   ```
5. Open a pull request describing the motivation and approach.

### Adding a New Op

A new op should land in roughly four steps. The pattern below is the contract every
op in the codebase follows.

1. **Declare** the op in the appropriate public header under `include/ctorch/`.
2. **Implement** the forward kernel in `src/` (CPU) and `kernels/` (CUDA), registering
   both with the dispatcher under the same op key.
3. **Register the backward** by deriving from `autograd::Function`, saving the inputs
   needed by the gradient, and computing input grads from the output grad.
4. **Add a parity test** under `tests/parity/` that compares the op's output and
   gradient against a PyTorch reference within an agreed tolerance.
<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
<!-- CONTACT -->
## Contact

Hayden &middot; [@Hayden727](https://github.com/Hayden727)

Project link: [https://github.com/Hayden727/ctorch](https://github.com/Hayden727/ctorch)
<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [PyTorch](https://pytorch.org/) and [libtorch](https://pytorch.org/cppdocs/) &mdash; the API this project deliberately mirrors.
- [tinygrad](https://github.com/tinygrad/tinygrad) and [micrograd](https://github.com/karpathy/micrograd) &mdash; for showing how compact a working autograd can be.
- [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler.html) &mdash; the trace-and-compile design behind Phase 5.
- [Eigen](https://eigen.tuxfamily.org/), [OpenBLAS](https://www.openblas.net/), [GoogleTest](https://github.com/google/googletest).
<!-- MARKDOWN LINKS &amp; IMAGES -->

<!-- Header shields -->
[cpp-shield]: https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white
[cpp-url]: https://en.cppreference.com/w/cpp/20
[cmake-shield]: https://img.shields.io/badge/CMake-%3E%3D_3.20-064F8C?style=for-the-badge&logo=cmake&logoColor=white
[cmake-url]: https://cmake.org/
[cuda-shield]: https://img.shields.io/badge/CUDA-12+-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cuda-url]: https://developer.nvidia.com/cuda-toolkit
[claude-shield]: https://img.shields.io/badge/Claude_Code-Powered-cc785c?style=for-the-badge&logo=anthropic&logoColor=white
[claude-url]: https://claude.ai/code

<!-- Built With shields -->
[cpp-built-shield]: https://img.shields.io/badge/C%2B%2B20-Core-00599C?style=for-the-badge&logo=cplusplus&logoColor=white
[cmake-built-shield]: https://img.shields.io/badge/CMake-Build-064F8C?style=for-the-badge&logo=cmake&logoColor=white
[openblas-shield]: https://img.shields.io/badge/OpenBLAS-CPU_BLAS-grey?style=for-the-badge
[openblas-url]: https://www.openblas.net/
[cuda-built-shield]: https://img.shields.io/badge/CUDA-Backend-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cublas-shield]: https://img.shields.io/badge/cuBLAS-GEMM-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cublas-url]: https://developer.nvidia.com/cublas
[gtest-shield]: https://img.shields.io/badge/GoogleTest-Unit-4285F4?style=for-the-badge&logo=google&logoColor=white
[gtest-url]: https://github.com/google/googletest
