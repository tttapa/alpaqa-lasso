#include <openmp/omp-problem.hpp>
#include <openmp/util.hpp>

namespace acl {

real_t OMPProblem::eval_f_grad_f(const real_t *x_, real_t *g_) const {
    cmtensor3 x{x_, n, p, q};
    mtensor3 g{g_, n, p, q};
    const std::array mat_mat{Eigen::IndexPair<index_t>{1, 0}};
    const std::array mat_tp_mat{Eigen::IndexPair<index_t>{0, 0}};
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
    for (index_t i = 0; i < q; ++i) {
        auto Ai  = data.A.chip(i, 2);
        auto Axi = work.Ax.chip(i, 2);
        auto xi  = x.chip(i, 2);
        auto bi  = data.b.chip(i, 2);
        auto gi  = g.chip(i, 2);
        Axi      = Ai.contract(xi, mat_mat) - bi;
        gi       = loss_scale * Ai.contract(Axi, mat_tp_mat) + λ_2 * xi;
        sq_norm += util::normSquared(Axi);
        sq_norm_x += util::normSquared(xi);
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