//===- src/linalg/blas_cuda.cu - cuBLAS GEMM wrapper ----------*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// `cublasSgemm` / `cublasDgemm` adapter. cuBLAS is column-major by
/// convention, but ctorch stores tensors row-major. The standard trick
/// is to compute `C^T = B^T A^T` — we feed cuBLAS the row-major buffers
/// directly with the operand order swapped, and the result lands in
/// row-major order with no extra transpose kernel.
///
/// The cuBLAS handle is cached per-thread per-device. Construction
/// hits a `cublasCreate` once on first use; everything after that is a
/// pointer load.
///
//===----------------------------------------------------------------------===//

#include "linalg/blas.h"

#include "ctorch/errors.h"

#include "cuda/device_guard.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <unordered_map>

namespace ctorch::linalg {

namespace {

// One cuBLAS handle per (thread, device). Created lazily.
struct ThreadLocalCublas {
    std::unordered_map<int, cublasHandle_t> handles;

    ~ThreadLocalCublas() {
        for (auto& [_, h] : handles) {
            cublasDestroy(h);
        }
    }

    cublasHandle_t get(int device_index) {
        auto it = handles.find(device_index);
        if (it != handles.end()) {
            return it->second;
        }
        cublasHandle_t h = nullptr;
        cublasStatus_t s = cublasCreate(&h);
        if (s != CUBLAS_STATUS_SUCCESS) {
            throw DeviceError("ctorch::matmul: cublasCreate failed");
        }
        handles.emplace(device_index, h);
        return h;
    }
};

cublasHandle_t cublas_for(int device_index) {
    static thread_local ThreadLocalCublas state;
    return state.get(device_index);
}

cublasOperation_t to_cublas(bool t) { return t ? CUBLAS_OP_T : CUBLAS_OP_N; }

void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw DeviceError(std::string("ctorch::matmul: ") + what + " failed with cuBLAS status " +
                          std::to_string(static_cast<int>(s)));
    }
}

} // namespace

// Row-major `C(M,N) = op(A)(M,K) * op(B)(K,N)`.
// cuBLAS sees column-major: the same buffers represent A^T and B^T.
// Computing `C^T = (op(B))^T * (op(A))^T` in cuBLAS column-major lays
// the result down row-major in C with no extra copies.
void gemm_cuda_f32(bool ta, bool tb, int M, int N, int K, float alpha, const float* a, int lda,
                   const float* b, int ldb, float beta, float* c, int ldc, int device_index) {
    cuda::DeviceGuard guard(device_index);
    cublasHandle_t h = cublas_for(device_index);
    check(cublasSgemm(h, to_cublas(tb), to_cublas(ta), N, M, K, &alpha, b, ldb, a, lda, &beta, c,
                      ldc),
          "cublasSgemm");
}

void gemm_cuda_f64(bool ta, bool tb, int M, int N, int K, double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta, double* c, int ldc, int device_index) {
    cuda::DeviceGuard guard(device_index);
    cublasHandle_t h = cublas_for(device_index);
    check(cublasDgemm(h, to_cublas(tb), to_cublas(ta), N, M, K, &alpha, b, ldb, a, lda, &beta, c,
                      ldc),
          "cublasDgemm");
}

} // namespace ctorch::linalg
