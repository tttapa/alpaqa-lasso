#include <complex-openmp/complex-convert.hpp>
#include <complex-openmp/complex-omp-problem.hpp>

namespace acl {

void ComplexOMPProblem::eval_hess_L_prod(const real_t *x_ [[maybe_unused]],
                                         const real_t *y_ [[maybe_unused]],
                                         real_t scale, const real_t *v_,
                                         real_t *Hv_) const {
    cmmat vr{v_, 2 * n, p * q};
    mmat Hvr{Hv_, 2 * n, p * q};
    auto v  = r2c(crmat{vr});
    auto Hv = r2c(rmat{Hvr});
#pragma omp parallel for
    for (index_t i = 0; i < q; ++i) {
        auto Ai       = data.A->middleCols(i * n, n);
        auto Axi      = work.Ax.middleCols(i * p, p);
        auto vi       = v.middleCols(i * p, p);
        auto Hvi      = Hv.middleCols(i * p, p);
        Axi.noalias() = Ai * vi;
        if (λ_2 != 0)
            Hvi.noalias() =
                scale * loss_scale * (Ai.adjoint() * Axi) + (scale * λ_2) * vi;
        else
            Hvi.noalias() = scale * loss_scale * (Ai.adjoint() * Axi);
    }
}

} // namespace acl
