#include <openmp/omp-problem.hpp>
#include <openmp/util.hpp>

#include <alpaqa/functions/l1-norm.hpp>
#include <alpaqa/functions/prox.hpp>

namespace acl {

real_t OMPProblem::eval_prox_grad_step(real_t γ, const real_t *x_,
                                       const real_t *grad_ψ_, real_t *x̂_,
                                       real_t *p_) const {
    alpaqa::functions::L1Norm<config_t> reg = λ_1;
    cmmat x{x_, n, p * q}, grad_ψ{grad_ψ_, n, p * q};
    mmat x̂{x̂_, n, p * q}, step{p_, n, p * q};
    real_t norm = 0;
#pragma omp parallel for reduction(+ : norm)
    for (index_t i = 0; i < p * q; i++)
        norm += alpaqa::prox_step(reg, x.col(i), grad_ψ.col(i), x̂.col(i),
                                  step.col(i), γ, -γ);
    return norm;
}

index_t OMPProblem::eval_inactive_indices_res_lna(real_t γ, const real_t *x_,
                                                  const real_t *grad_ψ_,
                                                  index_t *J_) const {
    real_t thres = γ * λ_1;
    index_t nJ   = 0;
    for (index_t i = 0; i < n * p * q; i++)
        if (λ_1 == 0 || std::abs(x_[i] - γ * grad_ψ_[i]) > thres)
            J_[nJ++] = i;
    return nJ;
}

} // namespace acl