#include "problem.hpp"

namespace acl {

void Problem::init_cuda() {
    handle         = acl::cublasUniqueCreate();
    ones           = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    minus_ones     = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    loss_scale_gpu = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    λ_2_gpu        = acl::cudaAlloc<real_t>(static_cast<size_t>(q));
    A_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(m * n * q));
    b_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(m * p * q));
    x_gpu          = acl::cudaAlloc<real_t>(static_cast<size_t>(n * p * q));
    Ax_gpu         = acl::cudaAlloc<real_t>(static_cast<size_t>(m * p * q));
    norms_gpu      = acl::cudaAlloc<real_t>(2);
    vec work       = vec::Ones(q);
    check(cublasSetVector(static_cast<int>(q), sizeof(real_t), work.data(), 1,
                          ones.get(), 1),
          "set +1");
    work = -vec::Ones(q);
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