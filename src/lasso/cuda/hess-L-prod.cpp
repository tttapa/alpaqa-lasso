#include <cuda/cuda-problem.hpp>

namespace acl {

void CUDAProblem::eval_hess_L_prod(const real_t *x_ [[maybe_unused]],
                                   const real_t *y_ [[maybe_unused]],
                                   real_t scale, const real_t *v_,
                                   real_t *Hv_) const {
    auto stream = acl::cudaStreamAlloc(cudaStreamNonBlocking);
    // x ← v
    check(cudaMemcpyAsync(gpu.work.x.get(), v_,
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyHostToDevice, stream.get()),
          "memcpy v");
    // Ax ← A v
    cublasSetStream(handle.get(), stream.get());
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
                                      /* β */ gpu.constants.zeros.get(),
                                      /* y */ gpu.work.Ax.get(),
                                      /* inc y */ 1,
                                      /* stride y */ static_cast<int>(m),
                                      /* batch count */ static_cast<int>(q)),
            "Ax ← A v");
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
                                      /* β */ gpu.constants.zeros.get(),
                                      /* C */ gpu.work.Ax.get(),
                                      /* ld C */ static_cast<int>(m),
                                      /* stride C */ static_cast<long long>(m) *
                                          static_cast<long long>(p),
                                      /* batch count */ static_cast<int>(q)),
            "Ax ← A v");
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
            "x ← AᴴA v - λ₂v");
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
            "x ← AᴴA v - λ₂v");
    if (scale != 1) {
        cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST);
        check(cublasDscal(handle.get(), static_cast<int>(n * p * q), &scale,
                          gpu.work.x.get(), 1),
              "cublasDscal");
    }
    check(cudaMemcpyAsync(Hv_, gpu.work.x.get(),
                          sizeof(real_t) * static_cast<size_t>(n * p * q),
                          cudaMemcpyDeviceToHost, stream.get()),
          "memcpy Hv");
    cudaStreamSynchronize(stream.get());
}

} // namespace acl