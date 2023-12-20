#include "problem.hpp"

namespace acl {

#if ACL_WITH_CUDA
real_t Problem::eval_f_grad_f_cuda_stream(const real_t *x_, real_t *g_,
                                          cudaStream_t stream) const {
    /*
        Stream 1                          Stream 2
            |
            |      -- initial_fork -->         |
            copy x                            copy Ax ← b
            |      ---- copy_done --->         |
            ‖x‖                               Ax - b
            |      ----- x_done ----->         |
            |      <---- Axb_done ----         |
        ‖Ax - b‖                          Aᴴ(sAx - b)
        copy ‖x‖, ‖Ax - b‖                  copy g
            |      <------ done ------         |
            |
    */
    auto stream2      = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    auto initial_fork = acl::cudaEventAlloc();
    check(cudaEventRecord(initial_fork.get(), stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream2.get(), initial_fork.get()),
          "cudaStreamWaitEvent");
    // x ← x
    check(cudaMemcpyAsync(x_gpu.get(), x_,
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyHostToDevice, stream),
          "memcpy x");
    // Ax ← b
    check(cudaMemcpyAsync(Ax_gpu.get(), b_gpu.get(),
                          sizeof(real_t) * static_cast<size_t>(m * p * q),
                          cudaMemcpyDeviceToDevice, stream2.get()),
          "Ax ← b");
    auto copy_done = acl::cudaEventAlloc();
    check(cudaEventRecord(copy_done.get(), stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream2.get(), copy_done.get()),
          "cudaStreamWaitEvent");
    // Ax ← A x - b
    cublasSetStream(handle.get(), stream2.get());
    cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
#if CUBLAS_VER_MAJOR >= 12
    if (p == 1)
        check(
            cublasDgemvStridedBatched(handle.get(),
                                      /* trans */ CUBLAS_OP_N,
                                      /* rows A */ static_cast<int>(m),
                                      /* cols A */ static_cast<int>(n),
                                      /* α */ ones.get(),
                                      /* A */ A_gpu.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* x */ x_gpu.get(),
                                      /* inc x */ 1,
                                      /* stride x */ static_cast<int>(n),
                                      /* β */ minus_ones.get(),
                                      /* y */ Ax_gpu.get(),
                                      /* inc y */ 1,
                                      /* stride y */ static_cast<int>(m),
                                      /* batch count */ static_cast<int>(q)),
            "Ax ← A x - b");
    else
#endif
        check(
            cublasDgemmStridedBatched(handle.get(),
                                      /* op a */ CUBLAS_OP_N,
                                      /* op b */ CUBLAS_OP_N,
                                      /* rows A */ static_cast<int>(m),
                                      /* cols B */ static_cast<int>(p),
                                      /* cols A */ static_cast<int>(n),
                                      /* α */ ones.get(),
                                      /* A */ A_gpu.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* B */ x_gpu.get(),
                                      /* ld B */ static_cast<int>(n),
                                      /* stride B */ static_cast<long long>(n) *
                                          static_cast<long long>(p),
                                      /* β */ minus_ones.get(),
                                      /* C */ Ax_gpu.get(),
                                      /* ld C */ static_cast<int>(m),
                                      /* stride C */ static_cast<long long>(m) *
                                          static_cast<long long>(p),
                                      /* batch count */ static_cast<int>(q)),
            "Ax ← A x - b");
    auto Axb_done = acl::cudaEventAlloc();
    check(cudaEventRecord(Axb_done.get(), stream2.get()), "cudaEventRecord");
    // ‖x‖
    cublasSetStream(handle.get(), stream);
    check(cublasDnrm2(handle.get(), static_cast<int>(n * p * q), x_gpu.get(), 1,
                      &norms_gpu[1]),
          "‖x‖");
    auto x_done = acl::cudaEventAlloc();
    check(cudaEventRecord(x_done.get(), stream), "cudaEventRecord");
    // ‖A x - b‖
    check(cudaStreamWaitEvent(stream, Axb_done.get()), "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream);
    check(cublasDnrm2(handle.get(), static_cast<int>(m * p * q), Ax_gpu.get(),
                      1, &norms_gpu[0]),
          "‖A x - b‖");
    // Copy norms
    std::array<real_t, 2> nrm_nrm_x;
    check(cudaMemcpyAsync(nrm_nrm_x.data(), norms_gpu.get(), sizeof(nrm_nrm_x),
                          cudaMemcpyDeviceToHost, stream),
          "memcpy");

    // x ← Aᴴ(A x - b)
    check(cudaStreamWaitEvent(stream2.get(), x_done.get()),
          "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream2.get());
#if CUBLAS_VER_MAJOR >= 12
    if (p == 1)
        check(
            cublasDgemvStridedBatched(handle.get(),
                                      /* trans */ CUBLAS_OP_C,
                                      /* rows A */ static_cast<int>(m),
                                      /* cols A */ static_cast<int>(n),
                                      /* α */ loss_scale_gpu.get(),
                                      /* A */ A_gpu.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* x */ Ax_gpu.get(),
                                      /* inc x */ 1,
                                      /* stride x */ static_cast<long long>(m),
                                      /* β */ λ_2_gpu.get(),
                                      /* y */ x_gpu.get(),
                                      /* inc y */ 1,
                                      /* stride y */ static_cast<int>(n),
                                      /* batch count */ static_cast<int>(q)),
            "x ← Aᴴ(A x - b)");
    else
#endif
        check(
            cublasDgemmStridedBatched(handle.get(),
                                      /* op A */ CUBLAS_OP_C,
                                      /* op B */ CUBLAS_OP_N,
                                      /* rows Aᴴ */ static_cast<int>(n),
                                      /* cols B */ static_cast<int>(p),
                                      /* cols Aᴴ */ static_cast<int>(m),
                                      /* α */ loss_scale_gpu.get(),
                                      /* A */ A_gpu.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* B */ Ax_gpu.get(),
                                      /* ld B */ static_cast<int>(m),
                                      /* stride B */ static_cast<long long>(m) *
                                          static_cast<long long>(p),
                                      /* β */ λ_2_gpu.get(),
                                      /* C */ x_gpu.get(),
                                      /* ld C */ static_cast<int>(n),
                                      /* stride C */ static_cast<long long>(n) *
                                          static_cast<long long>(p),
                                      /* batch count */ static_cast<int>(q)),
            "x ← Aᴴ(A x - b)");
    // Copy g
    check(cudaMemcpyAsync(g_, x_gpu.get(),
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyDeviceToHost, stream2.get()),
          "memcpy");
    auto done = acl::cudaEventAlloc();
    check(cudaEventRecord(done.get(), stream2.get()), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream, done.get()), "cudaStreamWaitEvent");
    cudaStreamSynchronize(stream);
    auto [nrm, nrm_x] = nrm_nrm_x;
    return 0.5 * loss_scale * nrm * nrm + 0.5 * λ_2 * nrm_x * nrm_x;
}

real_t Problem::eval_f_grad_f_cuda_stream_cached(const real_t *x_, real_t *g_,
                                                 cudaStream_t stream) const {
    /*
        Stream 1                      Stream 2                      Stream 3
            |
            |     -- initial_fork -->     |     -- initial_fork -->     |
          copy x                      g = s Aᵀb                       w = Aᵀb
            |     ---- copy_done --->     |     ---- copy_done --->     |
           ‖x‖                      g = λ_2 x + g                w = AᵀAx - 2Aᵀb
            |     <----- w_done -----     |     <----- w_done -----     |
           xᵀw                       g = s w + g
          copy norms                    copy g
            |     <------ done -------    |
            |
    */
    auto stream2      = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    auto stream3      = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    auto initial_fork = acl::cudaEventAlloc();
    check(cudaEventRecord(initial_fork.get(), stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream2.get(), initial_fork.get()),
          "cudaStreamWaitEvent");
    check(cudaStreamWaitEvent(stream3.get(), initial_fork.get()),
          "cudaStreamWaitEvent");
    // x ← x
    check(cudaMemcpyAsync(x_gpu.get(), x_,
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyHostToDevice, stream),
          "memcpy x");
    auto copy_done = acl::cudaEventAlloc();
    check(cudaEventRecord(copy_done.get(), stream), "cudaEventRecord");
    // g ← loss_scale Aᵀb
    check(cudaMemcpyAsync(g_gpu.get(), Aᵀb_gpu.get(),
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyDeviceToDevice, stream2.get()),
          "memcpy g");
    cublasSetStream(handle.get(), stream2.get());
    cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
    check(cublasDscal(handle.get(), static_cast<int>(n * p * q),
                      loss_scale_gpu.get(), g_gpu.get(), 1),
          "cublasDscal");
    // w ← Aᵀb
    check(cudaMemcpyAsync(w_gpu.get(), Aᵀb_gpu.get(),
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyDeviceToDevice, stream3.get()),
          "memcpy w");
    // ‖x‖
    cublasSetStream(handle.get(), stream);
    cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
    check(cublasDnrm2(handle.get(), static_cast<int>(n * p * q), x_gpu.get(), 1,
                      &norms_gpu[1]),
          "‖x‖");
    if (λ_2 != 0) {
        // g ← λ_2 x + g
        check(cudaStreamWaitEvent(stream2.get(), copy_done.get()),
              "cudaStreamWaitEvent");
        cublasSetStream(handle.get(), stream2.get());
        cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
        check(cublasDaxpy(handle.get(), static_cast<int>(n * p * q),
                          λ_2_gpu.get(), x_gpu.get(), 1, g_gpu.get(), 1),
              "g += λ_2 x");
    }
    // w ← AᵀA x - 2 Aᵀb
    check(cudaStreamWaitEvent(stream3.get(), copy_done.get()),
          "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream3.get());
    cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
    check(cublasDgemmStridedBatched(handle.get(),
                                    /* op a */ CUBLAS_OP_N,
                                    /* op b */ CUBLAS_OP_N,
                                    /* rows A */ static_cast<int>(n),
                                    /* cols B */ static_cast<int>(p),
                                    /* cols A */ static_cast<int>(n),
                                    /* α */ ones.get(),
                                    /* A */ AᵀA_gpu.get(),
                                    /* ld A */ static_cast<int>(n),
                                    /* stride A */ static_cast<long long>(n) *
                                        static_cast<long long>(n),
                                    /* B */ x_gpu.get(),
                                    /* ld B */ static_cast<int>(n),
                                    /* stride B */ static_cast<long long>(n) *
                                        static_cast<long long>(p),
                                    /* β */ minus_twos.get(),
                                    /* C */ w_gpu.get(),
                                    /* ld C */ static_cast<int>(n),
                                    /* stride C */ static_cast<long long>(n) *
                                        static_cast<long long>(p),
                                    /* batch count */ static_cast<int>(q)),
          "w ← AᵀA x - 2 Aᵀb");
    auto w_done = acl::cudaEventAlloc();
    check(cudaEventRecord(w_done.get(), stream3.get()), "cudaEventRecord");
    // wᵀx
    check(cudaStreamWaitEvent(stream, w_done.get()), "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream);
    cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
    check(cublasDdot(handle.get(), static_cast<int>(n * p * q), x_gpu.get(), 1,
                     w_gpu.get(), 1, &norms_gpu[0]),
          "wᵀx");
    // Copy norms
    std::array<real_t, 2> nrm_sq_nrm_x;
    check(cudaMemcpyAsync(nrm_sq_nrm_x.data(), norms_gpu.get(),
                          sizeof(nrm_sq_nrm_x), cudaMemcpyDeviceToHost, stream),
          "memcpy");
    // g ← loss_scale w + g
    check(cudaStreamWaitEvent(stream2.get(), w_done.get()),
          "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream2.get());
    cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_DEVICE);
    check(cublasDaxpy(handle.get(), static_cast<int>(n * p * q),
                      loss_scale_gpu.get(), w_gpu.get(), 1, g_gpu.get(), 1),
          "g ← loss_scale w + g");
    // Copy g
    check(cudaMemcpyAsync(g_, g_gpu.get(),
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyDeviceToHost, stream2.get()),
          "memcpy");
    auto done = acl::cudaEventAlloc();
    check(cudaEventRecord(done.get(), stream2.get()), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream, done.get()), "cudaStreamWaitEvent");
    cudaStreamSynchronize(stream);
    auto [nrm_sq, nrm_x] = nrm_sq_nrm_x;
    return 0.5 * loss_scale * (nrm_sq + bᵀb) + 0.5 * λ_2 * nrm_x * nrm_x;
}

real_t Problem::eval_f_grad_f_cuda(const real_t *x_, real_t *g_) const {
    auto stream = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    if (AᵀA_gpu)
        return eval_f_grad_f_cuda_stream_cached(x_, g_, stream.get());
    else
        return eval_f_grad_f_cuda_stream(x_, g_, stream.get());
}
#endif

} // namespace acl