#include <openmp/omp-problem.hpp>

namespace acl {

real_t OMPProblem::eval_f_grad_f(const real_t *x_, real_t *g_) const {
    cmmat x{x_, n, p * q};
    mmat g{g_, n, p * q};
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
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
        sq_norm += Axi.squaredNorm();
        if (λ_2 != 0)
            sq_norm_x += xi.squaredNorm();
    }
    return 0.5 * loss_scale * sq_norm + 0.5 * λ_2 * sq_norm_x;
}

/*

        sq_norm += bᵀb;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
        for (index_t i = 0; i < q; ++i) {
            auto AᵀAi = AᵀA.chip(i, 2);
            auto Aᵀbi = Aᵀb.chip(i, 2);
            auto xi   = x.chip(i, 2);
            auto wi   = w.chip(i, 2);
            auto gi   = g.chip(i, 2);
            wi        = AᵀAi.contract(xi, mat_mat) - 2 * Aᵀbi;
            sq_norm += dot(wi, xi);
            sq_norm_x += normSquared(xi);
            gi = loss_scale * (wi + Aᵀbi) + λ_2 * xi;
        }

*/

} // namespace acl