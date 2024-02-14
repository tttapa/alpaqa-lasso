#include <openmp/omp-problem.hpp>

namespace acl {

void OMPProblem::eval_hess_L(const real_t *x_ [[maybe_unused]],
                             const real_t *y_ [[maybe_unused]], real_t scale,
                             real_t *H_) const {
    mmat H{H_, n * p * q, n * p * q};
    H.setZero();
#pragma omp parallel for
    for (index_t i = 0; i < q; ++i) {
        auto Ai      = data.A->middleCols(i * n, n);
        auto Hi      = H.block(i * n * p, i * n * p, n, n);
        Hi.noalias() = scale * loss_scale * (Ai.adjoint() * Ai);
        if (λ_2 != 0)
            Hi += (scale * λ_2) * mat::Identity(n, n);
        for (index_t j = 1; j < p; ++j) {
            auto Hij = H.block(i * n * p + j * n, i * n * p + j * n, n, n);
            Hij      = Hi;
        }
    }
}

} // namespace acl