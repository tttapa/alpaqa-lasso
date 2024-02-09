#include <complex-openmp/complex-convert.hpp>
#include <complex-openmp/complex-omp-problem.hpp>

namespace acl {

real_t ComplexOMPProblem::eval_f(const real_t *x_) const {
    cmmat xr{x_, 2 * n, p * q};
    auto x           = r2c(crmat{xr});
    real_t sq_norm   = 0;
    real_t sq_norm_x = 0;
#pragma omp parallel for reduction(+ : sq_norm) reduction(+ : sq_norm_x)
    for (index_t i = 0; i < q; ++i) {
        auto Ai       = data.A->middleCols(i * n, n);
        auto Axi      = work.Ax.middleCols(i * p, p);
        auto xi       = x.middleCols(i * p, p);
        auto bi       = data.b->middleCols(i * p, p);
        Axi.noalias() = Ai * xi - bi;
        sq_norm += Axi.squaredNorm();
        if (λ_2 != 0)
            sq_norm_x += xi.squaredNorm();
    }
    return 0.5 * loss_scale * sq_norm + 0.5 * λ_2 * sq_norm_x;
}

} // namespace acl
