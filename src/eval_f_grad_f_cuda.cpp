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
    check(cudaMemcpyAsync(x_gpu.get(), x_, 2 * sizeof(real_t) * F * n,
                          cudaMemcpyHostToDevice, stream),
          "memcpy x");
    // Ax ← b
    check(cudaMemcpyAsync(Ax_gpu.get(), b_gpu.get(), 2 * sizeof(real_t) * m * F,
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
    check(cublasZgemvStridedBatched(
              handle.get(), CUBLAS_OP_N, static_cast<int>(m),
              static_cast<int>(n), ones.get(), A_gpu.get(), static_cast<int>(m),
              static_cast<long long>(m * n), x_gpu.get(), 1,
              static_cast<int>(n), minus_ones.get(), Ax_gpu.get(), 1,
              static_cast<int>(m), static_cast<int>(F)),
          "Ax ← A x - b");
#else
    check(
        cublasZgemmStridedBatched(handle.get(),
                                  /* op a */ CUBLAS_OP_N,
                                  /* op b */ CUBLAS_OP_N,
                                  /* rows A */ static_cast<int>(m),
                                  /* cols B */ 1,
                                  /* cols A */ static_cast<int>(n),
                                  /* α */ ones.get(),
                                  /* A */ A_gpu.get(),
                                  /* ld A */ static_cast<int>(m),
                                  /* stride A */ static_cast<long long>(m * n),
                                  /* B */ x_gpu.get(),
                                  /* ld B */ static_cast<int>(n),
                                  /* stride B */ static_cast<int>(n),
                                  /* β */ minus_ones.get(),
                                  /* C */ Ax_gpu.get(),
                                  /* ld C */ static_cast<int>(m),
                                  /* stride C */ static_cast<int>(m),
                                  /* batch count */ static_cast<int>(F)),
        "Ax ← A x - b");
#endif
    auto Axb_done = acl::cudaEventAlloc();
    check(cudaEventRecord(Axb_done.get(), stream2.get()), "cudaEventRecord");
    // ‖x‖
    cublasSetStream(handle.get(), stream);
    check(cublasDznrm2(handle.get(), static_cast<int>(n * F), x_gpu.get(), 1,
                       &norms_gpu[1]),
          "‖x‖");
    auto x_done = acl::cudaEventAlloc();
    check(cudaEventRecord(x_done.get(), stream), "cudaEventRecord");
    // ‖A x - b‖
    check(cudaStreamWaitEvent(stream, Axb_done.get()), "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream);
    check(cublasDznrm2(handle.get(), static_cast<int>(m * F), Ax_gpu.get(), 1,
                       &norms_gpu[0]),
          "‖A x - b‖");
    // Copy norms
    std::array<real_t, 2> nrm_nrmx;
    check(cudaMemcpyAsync(nrm_nrmx.data(), norms_gpu.get(), sizeof(nrm_nrmx),
                          cudaMemcpyDeviceToHost, stream),
          "memcpy");

    // x ← Aᴴ(A x - b)
    check(cudaStreamWaitEvent(stream2.get(), x_done.get()),
          "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream2.get());
#if CUBLAS_VER_MAJOR >= 12
    check(cublasZgemvStridedBatched(
              handle.get(), CUBLAS_OP_C, static_cast<int>(m),
              static_cast<int>(n), ones.get(), A_gpu.get(), static_cast<int>(m),
              static_cast<long long>(m * n), Ax_gpu.get(), 1,
              static_cast<int>(m), mus.get(), x_gpu.get(), 1,
              static_cast<int>(n), static_cast<int>(F)),
          "x ← Aᴴ(A x - b)");
#else
    check(
        cublasZgemmStridedBatched(handle.get(),
                                  /* op A */ CUBLAS_OP_C,
                                  /* op B */ CUBLAS_OP_N,
                                  /* rows Aᴴ */ static_cast<int>(n),
                                  /* cols B */ 1,
                                  /* cols Aᴴ */ static_cast<int>(m),
                                  /* α */ ones.get(),
                                  /* A */ A_gpu.get(),
                                  /* ld A */ static_cast<int>(m),
                                  /* stride A */ static_cast<long long>(m * n),
                                  /* B */ Ax_gpu.get(),
                                  /* ld B */ static_cast<int>(m),
                                  /* stride B */ static_cast<int>(m),
                                  /* β */ mus.get(),
                                  /* C */ x_gpu.get(),
                                  /* ld C */ static_cast<int>(n),
                                  /* stride C */ static_cast<int>(n),
                                  /* batch count */ static_cast<int>(F)),
        "x ← Aᴴ(A x - b)");
#endif
    // Copy g
    check(cudaMemcpyAsync(g_, x_gpu.get(), 2 * sizeof(real_t) * n * F,
                          cudaMemcpyDeviceToHost, stream2.get()),
          "memcpy");
    auto done = acl::cudaEventAlloc();
    check(cudaEventRecord(done.get(), stream2.get()), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream, done.get()), "cudaStreamWaitEvent");
    cudaStreamSynchronize(stream);
    auto [nrm, nrm_x] = nrm_nrmx;
    return real_t(0.5) * (nrm * nrm + μ * nrm_x * nrm_x);
}

real_t Problem::eval_f_grad_f_cuda(const real_t *x_, real_t *g_) const {
    auto stream = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    return eval_f_grad_f_cuda_stream(x_, g_, stream.get());
}
#endif

} // namespace acl