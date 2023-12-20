#include "cuda-util.hpp"
#include "problem.hpp"

namespace acl {

void Problem::init_cuda(bool cache_hessian) {
    handle         = acl::cublasUniqueCreate();
    zeros          = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    ones           = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    minus_ones     = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    minus_twos     = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    loss_scale_gpu = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    λ_2_gpu        = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    A_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(m * n * q));
    b_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(m * p * q));
    x_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(n * p * q));
    g_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(n * p * q));
    Ax_gpu         = acl::cudaAlloc<real_t>(static_cast<size_t>(m * p * q));
    norms_gpu      = acl::cudaAlloc<real_t>(2);
    AᵀA_gpu        = acl::cudaAlloc<real_t>(static_cast<size_t>(n * n * q));
    Aᵀb_gpu        = acl::cudaAlloc<real_t>(static_cast<size_t>(n * p * q));
    w_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(n * p * q));
    vec work(q);
    if (cache_hessian) {
        work.setZero(q);
        check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(),
                              1, zeros.get(), 1),
              "set 0");
        work.setConstant(q, -2);
        check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(),
                              1, minus_twos.get(), 1),
              "set -2");
    }
    work.setConstant(q, 1);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          ones.get(), 1),
          "set +1");
    work.setConstant(q, -1);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          minus_ones.get(), 1),
          "set -1");
    work.setConstant(q, λ_2);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          λ_2_gpu.get(), 1),
          "set λ_2");
    work.setConstant(q, loss_scale);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          loss_scale_gpu.get(), 1),
          "set loss_scale");
    auto stream1 = acl::cudaStreamAlloc(cudaStreamNonBlocking),
         stream2 = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    check(cudaMemcpyAsync(A_gpu.get(), A.data(),
                          sizeof(real_t) * static_cast<size_t>(m * n * q),
                          cudaMemcpyHostToDevice, stream1.get()),
          "memcpy A");
    check(cudaMemcpyAsync(b_gpu.get(), b.data(),
                          sizeof(real_t) * static_cast<size_t>(m * p * q),
                          cudaMemcpyHostToDevice, stream2.get()),
          "memcpy b");
    if (cache_hessian) {
        auto copy_A_done = acl::cudaEventAlloc();
        check(cudaEventRecord(copy_A_done.get(), stream1.get()),
              "cudaEventRecord");
        cublasSetStream(handle.get(), stream1.get());
        cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
        check(
            cublasDgemmStridedBatched(handle.get(),
                                      /* op A */ CUBLAS_OP_C,
                                      /* op B */ CUBLAS_OP_N,
                                      /* rows Aᴴ */ static_cast<int>(n),
                                      /* cols B */ static_cast<int>(n),
                                      /* cols Aᴴ */ static_cast<int>(m),
                                      /* α */ ones.get(),
                                      /* A */ A_gpu.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* B */ A_gpu.get(),
                                      /* ld B */ static_cast<int>(m),
                                      /* stride B */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* β */ zeros.get(),
                                      /* C */ AᵀA_gpu.get(),
                                      /* ld C */ static_cast<int>(n),
                                      /* stride C */ static_cast<long long>(n) *
                                          static_cast<long long>(n),
                                      /* batch count */ static_cast<int>(q)),
            "AᴴA");
        check(cudaStreamWaitEvent(stream2.get(), copy_A_done.get()),
              "cudaStreamWaitEvent");
        cublasSetStream(handle.get(), stream2.get());
        check(
            cublasDgemmStridedBatched(handle.get(),
                                      /* op A */ CUBLAS_OP_C,
                                      /* op B */ CUBLAS_OP_N,
                                      /* rows Aᴴ */ static_cast<int>(n),
                                      /* cols B */ static_cast<int>(p),
                                      /* cols Aᴴ */ static_cast<int>(m),
                                      /* α */ ones.get(),
                                      /* A */ A_gpu.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* B */ b_gpu.get(),
                                      /* ld B */ static_cast<int>(m),
                                      /* stride B */ static_cast<long long>(m) *
                                          static_cast<long long>(p),
                                      /* β */ zeros.get(),
                                      /* C */ Aᵀb_gpu.get(),
                                      /* ld C */ static_cast<int>(n),
                                      /* stride C */ static_cast<long long>(n) *
                                          static_cast<long long>(n),
                                      /* batch count */ static_cast<int>(q)),
            "Aᴴb");
    }
    cudaStreamSynchronize(stream1.get());
    cudaStreamSynchronize(stream2.get());
}

void Problem::update_λ_2_cuda() {
    if (!λ_2_gpu)
        return;
    vec work = vec::Constant(q, λ_2);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          λ_2_gpu.get(), 1),
          "set λ_2");
}

} // namespace acl