#include <complex-openmp/complex-convert.hpp>
#include <complex-openmp/complex-omp-problem.hpp>

namespace acl {

void ComplexOMPProblem::eval_grad_f(const real_t *x_, real_t *g_) const {
    cmmat xr{x_, 2 * n, p * q};
    mmat gr{g_, 2 * n, p * q};
    auto x = r2c(crmat{xr});
    auto g = r2c(rmat{gr});
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

} // namespace acl
