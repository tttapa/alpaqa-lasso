#include <openmp/omp-problem.hpp>
#include <openmp/util.hpp>

namespace acl {

void OMPProblem::eval_hess_L_prod(const real_t *x_ [[maybe_unused]],
                                  const real_t *y_ [[maybe_unused]],
                                  real_t scale, const real_t *v_,
                                  real_t *Hv_) const {
    cmtensor3 v{v_, n, p, q};
    mtensor3 Hv{Hv_, n, p, q};
    const std::array mat_mat{Eigen::IndexPair<index_t>{1, 0}};
    const std::array mat_tp_mat{Eigen::IndexPair<index_t>{0, 0}};
#pragma omp parallel for
    for (index_t i = 0; i < q; ++i) {
        auto Ai  = data.A.chip(i, 2);
        auto Axi = work.Ax.chip(i, 2);
        auto vi  = v.chip(i, 2);
        auto Hvi = Hv.chip(i, 2);
        Axi      = Ai.contract(vi, mat_mat);
        Hvi      = scale * loss_scale * Ai.contract(Axi, mat_tp_mat) +
              (scale * λ_2) * vi;
    }
}

} // namespace acl