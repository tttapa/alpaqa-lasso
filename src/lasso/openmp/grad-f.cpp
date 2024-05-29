#include <openmp/omp-problem.hpp>

namespace acl {

void OMPProblem::eval_grad_f(const real_t *x_, real_t *g_) const {
    cmmat x{x_, n, p * q};
    mmat g{g_, n, p * q};
#pragma omp parallel for
    for (index_t i = 0; i < q; ++i) {
        auto Ai       = data.A->middleCols(i * n, n);
        auto Axi      = work.Ax.middleCols(i * p, p);
        auto xi       = x.middleCols(i * p, p);
        auto bi       = data.b->middleCols(i * p, p);
        auto gi       = g.middleCols(i * p, p);
        Axi.noalias() = Ai * xi - bi;
        if (λ_2 != 0)
            gi.noalias() = loss_scale * (Ai.adjoint() * Axi) + λ_2 * xi;
        else
            gi.noalias() = loss_scale * (Ai.adjoint() * Axi);
    }
}

/*

#pragma omp parallel for
        for (index_t i = 0; i < q; ++i) {
            auto AᵀAi = AᵀA.chip(i, 2);
            auto Aᵀbi = Aᵀb.chip(i, 2);
            auto xi   = x.chip(i, 2);
            auto gi   = g.chip(i, 2);
            gi = loss_scale * AᵀAi.contract(xi, mat_mat) - loss_scale * Aᵀbi +
                 λ_2 * xi;
        }

*/

} // namespace acl