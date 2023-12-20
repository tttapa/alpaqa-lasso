#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <source_location>
#include <stdexcept>
#include <type_traits>

namespace acl {

inline const char *enum_name(cublasStatus_t s) {
    switch (s) { // clang-format off
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "<unknown>";
    } // clang-format on
}

inline void
check(cublasStatus_t status, const char *user_msg,
      const std::source_location &location = std::source_location::current()) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return;
    std::string msg = "cuBLAS error ";
    msg += enum_name(status);
    msg += " (";
    msg += std::to_string(static_cast<int>(status));
    msg += "): ‘";
    msg += user_msg;
    msg += "’ at ";
    msg += location.file_name();
    msg += ":";
    msg += std::to_string(location.line());
    msg += " in ";
    msg += location.function_name();
    throw std::runtime_error(msg);
}

inline void
check(cudaError status, const char *user_msg,
      const std::source_location &location = std::source_location::current()) {
    if (status == cudaSuccess)
        return;
    std::string msg = "CUDA error ";
    msg += ::cudaGetErrorString(status);
    msg += " (";
    msg += std::to_string(static_cast<int>(status));
    msg += "): ‘";
    msg += user_msg;
    msg += "’ at ";
    msg += location.file_name();
    msg += ":";
    msg += std::to_string(location.line());
    msg += " in ";
    msg += location.function_name();
    throw std::runtime_error(msg);
}

struct cudaDeleter {
    void operator()(void *p) const { check(::cudaFree(p), "cudaFree"); }
};
template <class T>
using cudaUniquePtr = std::unique_ptr<T[], cudaDeleter>;

template <class T>
auto cudaAlloc(size_t n) {
    T *p;
    size_t nbytes = n * sizeof(T);
    check(::cudaMalloc(reinterpret_cast<void **>(&p), nbytes), "cudaMalloc");
    return cudaUniquePtr<T> {p};
}

struct cudaEventDeleter {
    void operator()(cudaEvent_t e) const {
        check(::cudaEventDestroy(e), "cudaEventDestroy");
    }
};
using cudaUniqueEvent =
    std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cudaEventDeleter>;

inline auto cudaEventAlloc(unsigned flags = cudaEventDefault) {
    cudaEvent_t e;
    check(::cudaEventCreateWithFlags(&e, flags), "cudaEventCreateWithFlags");
    return cudaUniqueEvent {e};
}

struct cudaStreamDeleter {
    void operator()(cudaStream_t s) const {
        check(::cudaStreamDestroy(s), "cudaStreamDestroy");
    }
};
using cudaUniqueStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, cudaStreamDeleter>;

inline auto cudaStreamAlloc(unsigned flags = cudaStreamDefault) {
    cudaStream_t s;
    check(::cudaStreamCreateWithFlags(&s, flags), "cudaStreamCreateWithFlags");
    return cudaUniqueStream {s};
}

struct cublasDeleter {
    void operator()(cublasHandle_t h) const {
        check(::cublasDestroy(h), "cublasDestroy");
    }
};
using cublasUniqueHandle =
    std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, cublasDeleter>;

inline auto cublasUniqueCreate() {
    cublasHandle_t h;
    check(::cublasCreate(&h), "cublasCreate");
    return cublasUniqueHandle {h};
}

} // namespace acl