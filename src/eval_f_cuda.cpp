#include "problem.hpp"

namespace acl {

#if ACL_WITH_CUDA
real_t Problem::eval_f_cuda_stream(const real_t *x_,
                                   cudaStream_t stream) const {
    /*
        Stream 1                          Stream 2
            |
            |      -- initial_fork -->         |
            copy x                            copy Ax ← b
            |      ---- copy_done --->         |
            ‖x‖                               Ax - b
            |                              ‖Ax - b‖
            |      <------ done ------         |
        copy ‖x‖, ‖Ax - b‖
            |
    */
    auto stream2      = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    auto initial_fork = acl::cudaEventAlloc();
    check(cudaEventRecord(initial_fork.get(), stream), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream2.get(), initial_fork.get()),
          "cudaStreamWaitEvent");
    // x ← x
    check(cudaMemcpyAsync(x_gpu.get(), x_, 2 * sizeof(real_t) * q * n,
                          cudaMemcpyHostToDevice, stream),
          "memcpy x");
    // Ax ← b
    check(cudaMemcpyAsync(Ax_gpu.get(), b_gpu.get(), 2 * sizeof(real_t) * m * q,
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
              static_cast<int>(m), static_cast<int>(q)),
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
                                  /* batch count */ static_cast<int>(q)),
        "Ax ← A x - b");
#endif
    // ‖x‖
    cublasSetStream(handle.get(), stream);
    check(cublasDznrm2(handle.get(), static_cast<int>(n * q), x_gpu.get(), 1,
                       &norms_gpu[1]),
          "‖x‖");
    // ‖A x - b‖
    cublasSetStream(handle.get(), stream2.get());
    check(cublasDznrm2(handle.get(), static_cast<int>(m * q), Ax_gpu.get(), 1,
                       &norms_gpu[0]),
          "‖A x - b‖");
    auto done = acl::cudaEventAlloc();
    check(cudaEventRecord(done.get(), stream2.get()), "cudaEventRecord");
    check(cudaStreamWaitEvent(stream, done.get()), "cudaStreamWaitEvent");
    std::array<real_t, 2> nrm_nrmx;
    check(cudaMemcpyAsync(nrm_nrmx.data(), norms_gpu.get(), sizeof(nrm_nrmx),
                          cudaMemcpyDeviceToHost, stream),
          "memcpy");
    cudaStreamSynchronize(stream);
    auto [nrm, nrm_x] = nrm_nrmx;
    return real_t(0.5) * (nrm * nrm + μ * nrm_x * nrm_x);
}

real_t Problem::eval_f_cuda(const real_t *x_) const {
    auto stream = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    return eval_f_cuda_stream(x_, stream.get());
}
#endif

} // namespace acl