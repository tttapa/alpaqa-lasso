#include <complex-openmp/complex-convert.hpp>
#include <complex-openmp/complex-omp-problem.hpp>

#include <stdexcept>

#include <alpaqa/functions/l1-norm.hpp>
#include <alpaqa/functions/prox.hpp>

namespace acl {

real_t ComplexOMPProblem::eval_prox_grad_step(real_t γ, const real_t *x_,
                                              const real_t *grad_ψ_, real_t *x̂_,
                                              real_t *p_) const {
    alpaqa::functions::L1NormComplex<config_t, real_t> reg = λ_1;
    cmmat x{x_, 2 * n, p * q}, grad_ψ{grad_ψ_, 2 * n, p * q};
    mmat x̂{x̂_, 2 * n, p * q}, step{p_, 2 * n, p * q};
    real_t norm = 0;
#pragma omp parallel for reduction(+ : norm)
    for (index_t i = 0; i < p * q; i++)
        norm += alpaqa::prox_step(reg, x.col(i), grad_ψ.col(i), x̂.col(i),
                                  step.col(i), γ, -γ);
    return norm;
}

index_t ComplexOMPProblem::eval_inactive_indices_res_lna(
    [[maybe_unused]] real_t γ, [[maybe_unused]] const real_t *x_,
    [[maybe_unused]] const real_t *grad_ψ_,
    [[maybe_unused]] index_t *J_) const {
    throw std::logic_error(
        "ComplexOMPProblem::eval_inactive_indices_res_lna is not implemented");
}

} // namespace acl
