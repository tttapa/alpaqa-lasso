#include <cuda/cuda-problem.hpp>

namespace acl {

real_t CUDAProblem::eval_f_grad_f_cuda_stream(const real_t *x_, real_t *g_,
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
    check(cudaMemcpyAsync(gpu.work.x.get(), x_,
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyHostToDevice, stream),
          "memcpy x");
    // Ax ← b
    check(cudaMemcpyAsync(gpu.work.Ax.get(), gpu.data.b.get(),
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
                                      /* α */ gpu.constants.ones.get(),
                                      /* A */ gpu.data.A.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* x */ gpu.work.x.get(),
                                      /* inc x */ 1,
                                      /* stride x */ static_cast<int>(n),
                                      /* β */ gpu.constants.minus_ones.get(),
                                      /* y */ gpu.work.Ax.get(),
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
                                      /* α */ gpu.constants.ones.get(),
                                      /* A */ gpu.data.A.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* B */ gpu.work.x.get(),
                                      /* ld B */ static_cast<int>(n),
                                      /* stride B */ static_cast<long long>(n) *
                                          static_cast<long long>(p),
                                      /* β */ gpu.constants.minus_ones.get(),
                                      /* C */ gpu.work.Ax.get(),
                                      /* ld C */ static_cast<int>(m),
                                      /* stride C */ static_cast<long long>(m) *
                                          static_cast<long long>(p),
                                      /* batch count */ static_cast<int>(q)),
            "Ax ← A x - b");
    auto Axb_done = acl::cudaEventAlloc();
    check(cudaEventRecord(Axb_done.get(), stream2.get()), "cudaEventRecord");
    // ‖x‖
    cublasSetStream(handle.get(), stream);
    check(cublasDnrm2(handle.get(), static_cast<int>(n * p * q),
                      gpu.work.x.get(), 1, &gpu.work.norms[1]),
          "‖x‖");
    auto x_done = acl::cudaEventAlloc();
    check(cudaEventRecord(x_done.get(), stream), "cudaEventRecord");
    // ‖A x - b‖
    check(cudaStreamWaitEvent(stream, Axb_done.get()), "cudaStreamWaitEvent");
    cublasSetStream(handle.get(), stream);
    check(cublasDnrm2(handle.get(), static_cast<int>(m * p * q),
                      gpu.work.Ax.get(), 1, &gpu.work.norms[0]),
          "‖A x - b‖");
    // Copy norms
    std::array<real_t, 2> nrm_nrm_x;
    check(cudaMemcpyAsync(nrm_nrm_x.data(), gpu.work.norms.get(),
                          sizeof(nrm_nrm_x), cudaMemcpyDeviceToHost, stream),
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
                                      /* α */ gpu.constants.loss_scale.get(),
                                      /* A */ gpu.data.A.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* x */ gpu.work.Ax.get(),
                                      /* inc x */ 1,
                                      /* stride x */ static_cast<long long>(m),
                                      /* β */ gpu.constants.λ_2.get(),
                                      /* y */ gpu.work.x.get(),
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
                                      /* α */ gpu.constants.loss_scale.get(),
                                      /* A */ gpu.data.A.get(),
                                      /* ld A */ static_cast<int>(m),
                                      /* stride A */ static_cast<long long>(m) *
                                          static_cast<long long>(n),
                                      /* B */ gpu.work.Ax.get(),
                                      /* ld B */ static_cast<int>(m),
                                      /* stride B */ static_cast<long long>(m) *
                                          static_cast<long long>(p),
                                      /* β */ gpu.constants.λ_2.get(),
                                      /* C */ gpu.work.x.get(),
                                      /* ld C */ static_cast<int>(n),
                                      /* stride C */ static_cast<long long>(n) *
                                          static_cast<long long>(p),
                                      /* batch count */ static_cast<int>(q)),
            "x ← Aᴴ(A x - b)");
    // Copy g
    check(cudaMemcpyAsync(g_, gpu.work.x.get(),
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

real_t CUDAProblem::eval_f_grad_f(const real_t *x_, real_t *g_) const {
    auto stream = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    return eval_f_grad_f_cuda_stream(x_, g_, stream.get());
}

} // namespace acl